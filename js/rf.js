/**
 * Trains a Random Forest that maps features directly into a flat array structure
 * optimized for WebGPU Storage Buffers.
 * * WEBGPU STRUCT MIRROR:
 * struct Node {
 * feature_index: i32,  // If >= 0, check this feature index. If -1, this is a leaf.
 * threshold: f32,      // Split value
 * left_child: i32,     // Array index of left node
 * right_child: i32,    // Array index or negative representation of Leaf Class ID
 * };
 */
export class FlatRandomForest {
  constructor(options = {}) {
    this.numTrees = options.numTrees || 10;
    this.maxDepth = options.maxDepth || 10;
    this.minSamplesSplit = options.minSamplesSplit || 2;
    this.numClasses = options.numClasses || 2;

    this.forestBuffer = null; // Flat Float32Array containing all nodes
    this.treeRoots = null;    // Dynamic starting offsets for WebGPU uniforms
  }

  /**
   * Train the model on sparse painted coordinate features.
   */
  train(X, y, numFeatures) {
    if (!y || y.length === 0) {
      throw new Error("Training aborted: No labeled samples provided.");
    }
    const numSamples = y.length;
    this.treeRoots = new Int32Array(this.numTrees);

    // Estimate maximum possible nodes per tree to pre-allocate safely
    // Max nodes in a binary tree of depth D is 2^(D+1) - 1
    const maxNodesPerTree = Math.pow(2, this.maxDepth + 1) - 1;
    const slotsPerNode = 4;

    // Allocate a master ArrayBuffer for the entire forest execution
    const totalFloatSlots = this.numTrees * maxNodesPerTree * slotsPerNode;
    const masterBuffer = new ArrayBuffer(totalFloatSlots * 4);

    // Twin views over the exact same underlying binary storage layout
    const f32Forest = new Float32Array(masterBuffer);
    const i32Forest = new Int32Array(masterBuffer);

    let currentGlobalNodeOffset = 0;

    for (let t = 0; t < this.numTrees; t++) {
      // 1. Bootstrapping (Sample with replacement)
      const sampleIndices = new Int32Array(numSamples);
      if (this.numTrees === 1) {
        // Deterministic pass: Use all samples in order
        for (let i = 0; i < numSamples; i++) sampleIndices[i] = i;
      } else {
        // Standard random forest bootstrap
        for (let i = 0; i < numSamples; i++) {
            sampleIndices[i] = Math.floor(Math.random() * numSamples);
        }
      }

      // Record root index position for the shader configurations
      this.treeRoots[t] = currentGlobalNodeOffset;

      // 2. Track allocation offsets locally inside this tree build
      let localNodeCounter = 0;

      const buildNode = (indices, currentDepth) => {
        const nodeIdx = currentGlobalNodeOffset + localNodeCounter;
        localNodeCounter++;
        const slotOffset = nodeIdx * slotsPerNode;

        // Base check evaluations
        const isPure = this._checkPurity(y, indices);
        if (isPure || indices.length < this.minSamplesSplit || currentDepth >= this.maxDepth) {
          const leafClass = this._getMajorityClass(y, indices);
          // Leaf Node Rule: featIdx = -1, rightChild = -(class_id + 1)
          i32Forest[slotOffset + 0] = -1;
          f32Forest[slotOffset + 1] = 0.0;
          i32Forest[slotOffset + 2] = -1;
          i32Forest[slotOffset + 3] = -(leafClass + 1);
          return nodeIdx;
        }

        // Find optimal Gini threshold
        const split = this._findBestSplit(X, y, numFeatures, indices);

        if (split.featureIdx === -1) {
          const leafClass = this._getMajorityClass(y, indices);
          i32Forest[slotOffset + 0] = -1;
          f32Forest[slotOffset + 1] = 0.0;
          i32Forest[slotOffset + 2] = -1;
          i32Forest[slotOffset + 3] = -(leafClass + 1);
          return nodeIdx;
        }

        // Partition indices
        const leftList = [];
        const rightList = [];
        for (let i = 0; i < indices.length; i++) {
          const idx = indices[i];
          const val = X[idx * numFeatures + split.featureIdx];
          if (val < split.threshold) leftList.push(idx);
          else rightList.push(idx);
        }

        // Write split metadata to node slots early
        i32Forest[slotOffset + 0] = split.featureIdx;
        f32Forest[slotOffset + 1] = split.threshold;

        // Recursively compute branches
        const leftChildIdx = buildNode(new Int32Array(leftList), currentDepth + 1);
        const rightChildIdx = buildNode(new Int32Array(rightList), currentDepth + 1);

        // Backfill child array offset pointers smoothly
        i32Forest[slotOffset + 2] = leftChildIdx;
        i32Forest[slotOffset + 3] = rightChildIdx;

        return nodeIdx;
      };

      // Kickstart tree construction
      buildNode(sampleIndices, 0);
      currentGlobalNodeOffset += localNodeCounter;
    }

    // 3. Compact completely down into an optimized compact final buffer slice
    const totalActiveFloatSlots = currentGlobalNodeOffset * slotsPerNode;
    this.forestBuffer = new Float32Array(masterBuffer, 0, totalActiveFloatSlots);
  }

  _findBestSplit(X, y, numFeatures, indices) {
    let bestGiniGain = -1;
    let bestFeatureIdx = -1;
    let bestThreshold = 0;
    const currentGini = this._calculateGini(y, indices);

    for (let f = 0; f < numFeatures; f++) {
      const values = new Float32Array(indices.length);
      for (let i = 0; i < indices.length; i++) {
        values[i] = X[indices[i] * numFeatures + f];
      }
      values.sort();

      for (let i = 0; i < values.length - 1; i++) {
        if (values[i] === values[i + 1]) continue;
        const threshold = (values[i] + values[i + 1]) / 2.0;

        let leftCount = 0, rightCount = 0;
        const leftClassCounts = new Int32Array(this.numClasses);
        const rightClassCounts = new Int32Array(this.numClasses);

        for (let j = 0; j < indices.length; j++) {
          const idx = indices[j];
          const val = X[idx * numFeatures + f];
          const cls = y[idx];
          if (val < threshold) { leftClassCounts[cls]++; leftCount++; }
          else { rightClassCounts[cls]++; rightCount++; }
        }

        let giniLeftSum = 0, giniRightSum = 0;
        for (let c = 0; c < this.numClasses; c++) {
          if (leftCount > 0) { const p = leftClassCounts[c] / leftCount; giniLeftSum += p * p; }
          if (rightCount > 0) { const p = rightClassCounts[c] / rightCount; giniRightSum += p * p; }
        }

        const impurity = (leftCount / indices.length) * (1.0 - giniLeftSum) +
          (rightCount / indices.length) * (1.0 - giniRightSum);
        const gain = currentGini - impurity;

        if (gain > bestGiniGain) {
          bestGiniGain = gain;
          bestFeatureIdx = f;
          bestThreshold = threshold;
        }
      }
    }
    return { featureIdx: bestFeatureIdx, threshold: bestThreshold };
  }

  _calculateGini(y, indices) {
    const counts = new Int32Array(this.numClasses);
    for (let i = 0; i < indices.length; i++) counts[y[indices[i]]]++;
    let sumSq = 0;
    for (let c = 0; c < this.numClasses; c++) {
      const p = counts[c] / indices.length;
      sumSq += p * p;
    }
    return 1.0 - sumSq;
  }

  _checkPurity(y, indices) {
    const firstClass = y[indices[0]];
    for (let i = 1; i < indices.length; i++) {
      if (y[indices[i]] !== firstClass) return false;
    }
    return true;
  }

  _getMajorityClass(y, indices) {
    const counts = new Int32Array(this.numClasses);
    for (let i = 0; i < indices.length; i++) counts[y[indices[i]]]++;
    let maxCount = -1, majorityClass = 0;
    for (let c = 0; c < this.numClasses; c++) {
      if (counts[c] > maxCount) { maxCount = counts[c]; majorityClass = c; }
    }
    return majorityClass;
  }

  predictSingle(features) {
    const classVotes = new Float32Array(this.numClasses);
    const i32Forest = new Int32Array(this.forestBuffer.buffer, this.forestBuffer.byteOffset, this.forestBuffer.length);

    for (let t = 0; t < this.numTrees; t++) {
      let currentNodeIdx = this.treeRoots[t];
      while (true) {
        const slotOffset = currentNodeIdx * 4;
        const featIdx = i32Forest[slotOffset + 0];

        if (featIdx === -1) {
          const leafEncoded = i32Forest[slotOffset + 3];
          const classId = -leafEncoded - 1;
          classVotes[classId]++;
          break;
        }

        const threshold = this.forestBuffer[slotOffset + 1];
        if (features[featIdx] < threshold) {
          currentNodeIdx = i32Forest[slotOffset + 2];
        } else {
          currentNodeIdx = i32Forest[slotOffset + 3];
        }
      }
    }

    let maxVotes = -1, winningClass = 0;
    for (let c = 0; c < this.numClasses; c++) {
      if (classVotes[c] > maxVotes) { maxVotes = classVotes[c]; winningClass = c; }
    }
    return winningClass;
  }
}
