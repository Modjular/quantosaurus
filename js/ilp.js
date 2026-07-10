import { RF_CONFIG } from './config.js';

/**
 * Pure-JS writer for a minimal ilastik Pixel Classification `.ilp` project
 * (an HDF5 container) built from the app's `state`, with **no HDF5 library**
 * — the whole container is serialized by hand from first principles.
 *
 * This is possible because `.ilp` only needs a small, well-defined HDF5
 * subset, and libhdf5 (what desktop ilastik and h5py use to read it) is a
 * tolerant reader as long as the bytes are spec-compliant:
 *   - Every dataset is CONTIGUOUS, little-endian, unchunked, unfiltered.
 *   - Every string is FIXED-LENGTH, null-terminated — no HDF5 global heap.
 *   - Groups use the classic superblock-v0 "symbol table" layout (v1 B-tree +
 *     local heap + one symbol-table node per group), exactly what real .ilp
 *     files and h5py itself emit, so libhdf5 is guaranteed to accept it. No
 *     checksums required (unlike the modern v2/v3 superblock layout).
 *
 * Group naming (`lane{:04d}`, `labels{:03d}`, `block{:04d}`) and the raw-data
 * axis convention (`zyx`, shape `[1, height, width]` for a 2D image) mirror
 * ilastik's own serializer (`dataSelectionSerializer.py`,
 * `pixelClassificationSerializer.py`, `slotSerializer.py`), not just its file
 * format doc, so multi-image projects round-trip the same way desktop
 * ilastik itself would produce them.
 *
 * Every image's raw pixels are embedded directly in the `.ilp` — browsers
 * never expose a real filesystem path for an uploaded file (only a bare
 * filename), so a `FileSystem`-location reference (what desktop ilastik
 * itself writes for on-disk images) would only resolve if the original file
 * happened to still be sitting next to the exported `.ilp`. Instead each lane
 * uses ilastik's own "copy into project" representation
 * (`ProjectInternalDatasetInfo` in `opDataSelection.py`): the array lives at
 * `Input Data/local_data/<datasetId>` and the lane's `Raw Data` info group
 * points at it via `location: "ProjectInternal"` + `inner_path` — the exact
 * dispatch key `dataSelectionSerializer.py._readDatasetInfo` switches on
 * (`__class__` when present, else `location`) to decide how to load a lane.
 * The result is a single, fully self-contained, portable `.ilp` file.
 *
 * What is deliberately NOT included: a trained classifier
 * (`PixelClassification/ClassifierForests`). Quantosaurus trains its
 * `FlatRandomForest` on a custom on-GPU Gaussian-derivative feature bank
 * (`NUM_FEATURES` features — see backends/*.js), which does not correspond
 * 1:1 with the vigra features desktop ilastik computes itself. Embedding our
 * forest under a vigra-shaped `ClassifierForests` group would silently
 * mispredict. Per the format spec, `ClassifierForests` is optional — ilastik
 * simply retrains from the exported labels using its own features.
 *
 * Only the top-level `buildIlpProject` is meant for external use; everything
 * else here is private plumbing (a tiny HDF5-v0 writer core, then the
 * ilastik-specific tree assembly).
 */

// ---------------------------------------------------------------------------
// HDF5 constants for this writer's chosen configuration
// ---------------------------------------------------------------------------
const SIG = Uint8Array.of(0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a); // \x89HDF\r\n\x1a\n
const UNDEF = 0xffffffffffffffffn; // "undefined address" (all bits set)
// Group node "K": each group here is stored as a SINGLE symbol-table node, so
// every group must fit within 2*K_LEAF entries. Loaded-image lanes/label-sets
// scale with state.images.length; buildIlpProject throws a clear error if a
// project would exceed this rather than silently truncating/corrupting.
const K_LEAF = 32; // symbol-table node capacity = 2*K_LEAF = 64 entries
const K_INT = 32; // B-tree node capacity     = 2*K_INT  = 64 children

const pad8 = (n) => Math.ceil(n / 8) * 8;
const textEncoder = new TextEncoder();

// ---------------------------------------------------------------------------
// Growable little-endian byte sink with an 8-byte-aligned bump allocator.
// alloc(size) reserves a zero-filled, 8-aligned region and returns its file
// offset; everything is then written by absolute offset. `Uint8Array` is
// always zero-initialized on construction, so newly grown/allocated regions
// need no manual clearing.
// ---------------------------------------------------------------------------
class Sink {
  constructor() {
    this.buf = new Uint8Array(1 << 16);
    this.view = new DataView(this.buf.buffer);
    this.len = 0;
  }
  _ensure(n) {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length;
    while (cap < this.len + n) cap *= 2;
    const nb = new Uint8Array(cap);
    nb.set(this.buf.subarray(0, this.len));
    this.buf = nb;
    this.view = new DataView(this.buf.buffer);
  }
  alloc(size) {
    const off = this.len;
    const rounded = pad8(size);
    this._ensure(rounded);
    this.len += rounded;
    return off;
  }
  u8(o, v) { this.view.setUint8(o, v & 0xff); }
  u16(o, v) { this.view.setUint16(o, v & 0xffff, true); }
  u32(o, v) { this.view.setUint32(o, v >>> 0, true); }
  u64(o, v) { this.view.setBigUint64(o, typeof v === 'bigint' ? v : BigInt(v), true); }
  put(o, bytes) { this.buf.set(bytes, o); }
}

// Small standalone-buffer helper for the message-body builders below (they
// return a finished Uint8Array to be embedded as a message "body" later).
function newBuf(size) {
  const b = new Uint8Array(size);
  return { b, v: new DataView(b.buffer) };
}

// ---------------------------------------------------------------------------
// Datatype messages (message type 0x0003), version 1
// ---------------------------------------------------------------------------
// Fixed-length string: class 3, null-terminated, ASCII/UTF-8. Null-terminate
// (pad type 0) tells every reader to stop at the first NUL, so shorter
// entries in a fixed-width string array come back clean.
function dtStr(size) {
  const { b, v } = newBuf(8);
  v.setUint8(0, (1 << 4) | 3); // version 1, class 3 (string)
  v.setUint8(1, 0x00); // bit field: pad type 0 (null terminate), charset 0 (ASCII)
  v.setUint32(4, size, true); // element size in bytes
  return b;
}
// Fixed-point integer: class 0, little-endian, optional 2's-complement sign.
function dtInt(size, signed) {
  const { b, v } = newBuf(12);
  v.setUint8(0, (1 << 4) | 0); // version 1, class 0 (fixed-point)
  v.setUint8(1, signed ? 0x08 : 0x00); // bit0 order=LE, bit3 signedness
  v.setUint32(4, size, true); // element size in bytes
  v.setUint16(8, 0, true); // bit offset
  v.setUint16(10, size * 8, true); // bit precision
  return b;
}
// IEEE-754 little-endian double: class 1.
function dtFloat64() {
  const { b, v } = newBuf(20);
  v.setUint8(0, (1 << 4) | 1); // version 1, class 1 (floating-point)
  v.setUint8(1, 0x20); // bit field byte0: mantissa normalization = 2 (implied MSB)
  v.setUint8(2, 0x3f); // bit field byte1: sign bit location = 63
  v.setUint8(3, 0x00);
  v.setUint32(4, 8, true); // element size
  v.setUint16(8, 0, true); // bit offset
  v.setUint16(10, 64, true); // bit precision
  v.setUint8(12, 52); // exponent location
  v.setUint8(13, 11); // exponent size
  v.setUint8(14, 0); // mantissa location
  v.setUint8(15, 52); // mantissa size
  v.setUint32(16, 1023, true); // exponent bias
  return b;
}
// IEEE-754 little-endian single: class 1. Used for raw image pixel data —
// intensityArray is already Float32Array, so this stores it losslessly with
// no widening to float64.
function dtFloat32() {
  const { b, v } = newBuf(20);
  v.setUint8(0, (1 << 4) | 1); // version 1, class 1 (floating-point)
  v.setUint8(1, 0x20); // bit field byte0: mantissa normalization = 2 (implied MSB)
  v.setUint8(2, 0x1f); // bit field byte1: sign bit location = 31
  v.setUint8(3, 0x00);
  v.setUint32(4, 4, true); // element size
  v.setUint16(8, 0, true); // bit offset
  v.setUint16(10, 32, true); // bit precision
  v.setUint8(12, 23); // exponent location
  v.setUint8(13, 8); // exponent size
  v.setUint8(14, 0); // mantissa location
  v.setUint8(15, 23); // mantissa size
  v.setUint32(16, 127, true); // exponent bias
  return b;
}

// Dataspace message (0x0001), version 1. dims == [] -> scalar (rank 0).
function dataspace(dims) {
  const rank = dims.length;
  const { b, v } = newBuf(8 + rank * 8);
  v.setUint8(0, 1); // version
  v.setUint8(1, rank); // dimensionality
  v.setUint8(2, 0); // flags (no max dims)
  for (let i = 0; i < rank; i++) v.setBigUint64(8 + i * 8, BigInt(dims[i]), true);
  return b;
}

// Data Layout message (0x0008), version 3, class 1 (contiguous).
function layoutContiguous(addr, size) {
  const { b, v } = newBuf(18);
  v.setUint8(0, 3); // version
  v.setUint8(1, 1); // class 1 = contiguous
  v.setBigUint64(2, BigInt(addr), true); // data address
  v.setBigUint64(10, BigInt(size), true); // data size
  return b;
}

// Symbol Table message (0x0011): pointers to a group's B-tree + local heap.
function symbolTableMsg(btreeAddr, heapAddr) {
  const { b, v } = newBuf(16);
  v.setBigUint64(0, BigInt(btreeAddr), true);
  v.setBigUint64(8, BigInt(heapAddr), true);
  return b;
}

// Attribute message (0x000C), version 1. In v1 the name/datatype/dataspace
// sub-blocks are each padded to 8 bytes; the value follows unpadded.
function attributeMsg(name, dtBody, dsBody, valueBytes) {
  const nameBuf = textEncoder.encode(name + '\0');
  const nameP = pad8(nameBuf.length);
  const dtP = pad8(dtBody.length);
  const dsP = pad8(dsBody.length);
  const { b, v } = newBuf(8 + nameP + dtP + dsP + valueBytes.length);
  v.setUint8(0, 1); // version
  v.setUint8(1, 0); // reserved
  v.setUint16(2, nameBuf.length, true); // name size (incl. NUL)
  v.setUint16(4, dtBody.length, true); // datatype message size (unpadded)
  v.setUint16(6, dsBody.length, true); // dataspace message size (unpadded)
  let o = 8;
  b.set(nameBuf, o); o += nameP;
  b.set(dtBody, o); o += dtP;
  b.set(dsBody, o); o += dsP;
  b.set(valueBytes, o);
  return b;
}

// ---------------------------------------------------------------------------
// Version-1 object header. messages: [{ type, body:Uint8Array }]. Each message
// is framed with an 8-byte prefix and its data padded to a multiple of 8.
// ---------------------------------------------------------------------------
function writeObjectHeader(sink, messages) {
  const framed = messages.map((m) => ({ type: m.type, body: m.body, pad: pad8(m.body.length) }));
  const msgsSize = framed.reduce((s, m) => s + 8 + m.pad, 0);
  const addr = sink.alloc(16 + msgsSize); // 12-byte prefix + 4 pad -> 16
  sink.u8(addr + 0, 1); // version
  sink.u8(addr + 1, 0); // reserved
  sink.u16(addr + 2, messages.length); // number of header messages
  sink.u32(addr + 4, 1); // object reference count
  sink.u32(addr + 8, msgsSize); // header size (message data bytes)
  // addr+12..15 : alignment padding (already zero)
  let o = addr + 16;
  for (const m of framed) {
    sink.u16(o + 0, m.type); // message type
    sink.u16(o + 2, m.pad); // size of message data (padded)
    sink.u8(o + 4, 0); // flags
    // o+5..7 reserved (zero)
    sink.put(o + 8, m.body);
    o += 8 + m.pad;
  }
  return addr;
}

// ---------------------------------------------------------------------------
// Old-style group storage: local heap of names + a single-node v1 B-tree +
// (if non-empty) one symbol table node. Returns { btreeAddr, heapAddr }.
// ---------------------------------------------------------------------------
function writeGroupStorage(sink, entries) {
  if (entries.length > 2 * K_LEAF) {
    throw new Error(
      `ilp.js: group has ${entries.length} entries, exceeding the single symbol-table-node ` +
      `capacity of ${2 * K_LEAF}. Export fewer images per .ilp, or extend writeGroupStorage to ` +
      'split across multiple B-tree leaves.'
    );
  }
  // Symbol table node entries must be sorted by name (libhdf5 binary-searches).
  // All names used by this writer are plain ASCII identifiers, so JS string
  // comparison matches HDF5's required bytewise comparison.
  const sorted = entries.slice().sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  // --- Local heap: reserve [0,8) as the empty string (heap offset 0), then
  //     lay out each NUL-terminated name 8-byte aligned, then a free block. ---
  const nameOffset = new Map();
  const chunks = [];
  let cur = 8;
  for (const e of sorted) {
    const nb = textEncoder.encode(e.name + '\0');
    nameOffset.set(e.name, cur);
    chunks.push({ off: cur, buf: nb });
    cur += pad8(nb.length);
  }
  const freeOff = cur;
  const freeSize = 16; // minimal free block: [next=1][size]
  const dataSegSize = freeOff + freeSize;

  const heapHdr = sink.alloc(32);
  const dataSeg = sink.alloc(dataSegSize);
  sink.put(heapHdr, textEncoder.encode('HEAP'));
  sink.u8(heapHdr + 4, 0); // version
  sink.u64(heapHdr + 8, dataSegSize); // data segment size
  sink.u64(heapHdr + 16, freeOff); // offset to head of free list
  sink.u64(heapHdr + 24, dataSeg); // address of data segment
  for (const c of chunks) sink.put(dataSeg + c.off, c.buf);
  sink.u64(dataSeg + freeOff, 1n); // free block: next = 1 (sentinel: none)
  sink.u64(dataSeg + freeOff + 8, freeSize); // free block: size

  // --- Symbol table node (only when the group has members) ---
  let snodAddr = 0;
  if (sorted.length > 0) {
    const snodSize = 8 + 2 * K_LEAF * 40;
    snodAddr = sink.alloc(snodSize);
    sink.put(snodAddr, textEncoder.encode('SNOD'));
    sink.u8(snodAddr + 4, 1); // version
    sink.u16(snodAddr + 6, sorted.length); // number of symbols
    let so = snodAddr + 8;
    for (const e of sorted) {
      sink.u64(so + 0, nameOffset.get(e.name)); // link name offset (into heap)
      sink.u64(so + 8, e.ohAddr); // object header address
      sink.u32(so + 16, 0); // cache type 0 (no scratch-pad caching)
      // so+20..23 reserved, so+24..39 scratch-pad (all zero)
      so += 40;
    }
  }

  // --- Single-node v1 B-tree (node type 0 = group), level 0 ---
  const btreeSize = 24 + (2 * K_INT + 1) * 8 + 2 * K_INT * 8;
  const btree = sink.alloc(btreeSize);
  sink.put(btree, textEncoder.encode('TREE'));
  sink.u8(btree + 4, 0); // node type: group
  sink.u8(btree + 5, 0); // node level: leaf
  sink.u16(btree + 6, sorted.length > 0 ? 1 : 0); // entries used (0 or 1 child)
  sink.u64(btree + 8, UNDEF); // left sibling
  sink.u64(btree + 16, UNDEF); // right sibling
  // Keys/children interleave from btree+24: Key0, Child0, Key1, ...
  sink.u64(btree + 24, 0); // Key0 = heap offset 0 (empty string, <= all names)
  if (sorted.length > 0) {
    sink.u64(btree + 32, snodAddr); // Child0 -> the symbol table node
    const maxName = sorted[sorted.length - 1].name; // greatest name (bytewise)
    sink.u64(btree + 40, nameOffset.get(maxName)); // Key1 >= all names
  }

  return { btreeAddr: btree, heapAddr: heapHdr };
}

// ---------------------------------------------------------------------------
// Tree model. Nodes are plain objects:
//   dataset: { kind:'dataset', name, dtBody, dims, data:Uint8Array, attrs:[Uint8Array] }
//   group:   { kind:'group',   name, children:[node] }
// writeNode emits a node post-order and returns its object-header address.
// ---------------------------------------------------------------------------
function writeNode(sink, node) {
  if (node.kind === 'dataset') {
    // Per spec, an unallocated contiguous dataset's address is UNDEF, not 0
    // (0 is a real, meaningful file offset — the superblock's own location).
    const dataAddr = node.data.length > 0 ? sink.alloc(node.data.length) : UNDEF;
    if (node.data.length > 0) sink.put(dataAddr, node.data);
    const messages = [
      { type: 0x0003, body: node.dtBody },
      { type: 0x0001, body: dataspace(node.dims) },
      { type: 0x0008, body: layoutContiguous(dataAddr, node.data.length) },
    ];
    for (const a of node.attrs || []) messages.push({ type: 0x000c, body: a });
    return writeObjectHeader(sink, messages);
  }
  // group
  const entries = node.children.map((c) => ({ name: c.name, ohAddr: writeNode(sink, c) }));
  const { btreeAddr, heapAddr } = writeGroupStorage(sink, entries);
  return writeObjectHeader(sink, [{ type: 0x0011, body: symbolTableMsg(btreeAddr, heapAddr) }]);
}

// Serializes a root-level children list into a complete HDF5-v0 file.
function serializeHdf5(rootChildren) {
  const sink = new Sink();
  const sbAddr = sink.alloc(96);
  const entries = rootChildren.map((c) => ({ name: c.name, ohAddr: writeNode(sink, c) }));
  const rootStab = writeGroupStorage(sink, entries);
  const rootOH = writeObjectHeader(sink, [
    { type: 0x0011, body: symbolTableMsg(rootStab.btreeAddr, rootStab.heapAddr) },
  ]);

  // Superblock version 0.
  sink.put(sbAddr, SIG);
  sink.u8(sbAddr + 8, 0); // superblock version
  sink.u8(sbAddr + 9, 0); // free-space storage version
  sink.u8(sbAddr + 10, 0); // root group symbol table entry version
  sink.u8(sbAddr + 11, 0); // reserved
  sink.u8(sbAddr + 12, 0); // shared header message format version
  sink.u8(sbAddr + 13, 8); // size of offsets
  sink.u8(sbAddr + 14, 8); // size of lengths
  sink.u8(sbAddr + 15, 0); // reserved
  sink.u16(sbAddr + 16, K_LEAF); // group leaf node K
  sink.u16(sbAddr + 18, K_INT); // group internal node K
  sink.u32(sbAddr + 20, 0); // file consistency flags
  sink.u64(sbAddr + 24, 0); // base address
  sink.u64(sbAddr + 32, UNDEF); // free-space info address
  sink.u64(sbAddr + 40, BigInt(sink.len)); // end-of-file address
  sink.u64(sbAddr + 48, UNDEF); // driver information block address
  // Root group symbol table entry (cache type 1: scratch-pad holds B-tree/heap).
  sink.u64(sbAddr + 56, 0); // link name offset
  sink.u64(sbAddr + 64, rootOH); // object header address
  sink.u32(sbAddr + 72, 1); // cache type = 1
  sink.u32(sbAddr + 76, 0); // reserved
  sink.u64(sbAddr + 80, rootStab.btreeAddr); // scratch-pad: B-tree address
  sink.u64(sbAddr + 88, rootStab.heapAddr); // scratch-pad: local heap address

  // A zero-copy view trimmed to the exact byte length — Blob() (the only real
  // caller) reads an ArrayBufferView's byte range directly, so there's no need
  // to copy out of the (possibly over-allocated, post-growth) sink buffer here.
  return sink.buf.subarray(0, sink.len);
}

// ---------------------------------------------------------------------------
// Node constructors
// ---------------------------------------------------------------------------
function bytesFixed(bytes) {
  // Fixed-length payload: exact bytes, size == length (min 1, since a 0-byte
  // fixed-length string datatype isn't meaningful).
  const size = Math.max(bytes.length, 1);
  const data = new Uint8Array(size);
  data.set(bytes.subarray(0, Math.min(bytes.length, size)));
  return { size, data };
}
function dsBytes(name, bytes) {
  const { size, data } = bytesFixed(bytes);
  return { kind: 'dataset', name, dtBody: dtStr(size), dims: [], data };
}
function dsStr(name, s) {
  return dsBytes(name, textEncoder.encode(s));
}
function dsStrArray(name, list) {
  const encoded = list.map((s) => textEncoder.encode(s));
  const size = Math.max(1, ...encoded.map((e) => e.length));
  const data = new Uint8Array(size * list.length);
  encoded.forEach((e, i) => data.set(e.subarray(0, Math.min(e.length, size)), i * size));
  return { kind: 'dataset', name, dtBody: dtStr(size), dims: [list.length], data };
}
function dsInt64Scalar(name, n) {
  const data = new Uint8Array(8);
  new DataView(data.buffer).setBigInt64(0, BigInt(n), true);
  return { kind: 'dataset', name, dtBody: dtInt(8, true), dims: [], data };
}
function dsInt64Array(name, nums) {
  const data = new Uint8Array(8 * nums.length);
  const v = new DataView(data.buffer);
  nums.forEach((n, i) => v.setBigInt64(i * 8, BigInt(n), true));
  return { kind: 'dataset', name, dtBody: dtInt(8, true), dims: [nums.length], data };
}
function dsFloat64Array(name, nums) {
  const data = new Uint8Array(8 * nums.length);
  const v = new DataView(data.buffer);
  nums.forEach((n, i) => v.setFloat64(i * 8, n, true));
  return { kind: 'dataset', name, dtBody: dtFloat64(), dims: [nums.length], data };
}
// Raw N-D float32 payload — a zero-copy byte view over an existing
// Float32Array (e.g. an image's intensityArray), used for embedded pixel
// data. Every mainstream JS engine is little-endian, matching HDF5's
// required byte order here, so no repacking is needed.
function dsFloat32Nd(name, dims, floatArray) {
  const data = new Uint8Array(floatArray.buffer, floatArray.byteOffset, floatArray.byteLength);
  return { kind: 'dataset', name, dtBody: dtFloat32(), dims, data };
}
function dsInt32Matrix(name, rows) {
  const R = rows.length, C = rows[0].length;
  const data = new Uint8Array(4 * R * C);
  const v = new DataView(data.buffer);
  let o = 0;
  for (const row of rows) for (const val of row) { v.setInt32(o, val, true); o += 4; }
  return { kind: 'dataset', name, dtBody: dtInt(4, true), dims: [R, C], data };
}
// Booleans are stored as int8 (0/1), not an HDF5 enum. Nothing in the format
// spec or ilastik's own reader distinguishes the two for these fields.
function dsBoolScalar(name, val) {
  return { kind: 'dataset', name, dtBody: dtInt(1, true), dims: [], data: Uint8Array.of(val ? 1 : 0) };
}
function dsBoolArray(name, arr) {
  const data = new Uint8Array(arr.length);
  arr.forEach((val, i) => { data[i] = val ? 1 : 0; });
  return { kind: 'dataset', name, dtBody: dtInt(1, true), dims: [arr.length], data };
}
function dsBoolMatrix(name, rows) {
  const R = rows.length, C = rows[0].length;
  const data = new Uint8Array(R * C);
  let o = 0;
  for (const row of rows) for (const val of row) data[o++] = val ? 1 : 0;
  return { kind: 'dataset', name, dtBody: dtInt(1, true), dims: [R, C], data };
}
// Real (non-fill) uint8 payload, used for label blocks.
function dsUint8(name, dims, data, attrs) {
  return { kind: 'dataset', name, dtBody: dtInt(1, false), dims, data, attrs };
}
function group(name, children) {
  return { kind: 'group', name, children };
}
function strAttr(name, s) {
  const { size, data } = bytesFixed(textEncoder.encode(s));
  return attributeMsg(name, dtStr(size), dataspace([]), data);
}

// ---------------------------------------------------------------------------
// ilastik-specific helpers
// ---------------------------------------------------------------------------

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/**
 * Formats a Date like Python's `time.ctime()` (e.g. "Wed Aug 23 10:29:38
 * 2023"), matching the format real ilastik writes into its root `time`
 * dataset. Purely informational either way — ilastik's project loader
 * (`projectManager.py`) only ever writes this field, never parses it back.
 */
function formatCtime(date) {
  const pad2 = (n) => String(n).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, ' ');
  return `${WEEKDAYS[date.getDay()]} ${MONTHS[date.getMonth()]} ${day} ` +
    `${pad2(date.getHours())}:${pad2(date.getMinutes())}:${pad2(date.getSeconds())} ${date.getFullYear()}`;
}

/** Axis-tags JSON, matching ilastik's `vigra.AxisTags`-derived format. */
function makeAxistags(keys) {
  const typeFlags = { t: 4, z: 2, y: 2, x: 2, c: 1 };
  const axes = [...keys].map((key) => ({
    key,
    typeFlags: typeFlags[key] ?? 2,
    resolution: 0.0,
    description: '',
  }));
  return JSON.stringify({ axes }, null, 2);
}

// Precomputed once — every lane's Raw Data and every label block reuse the
// same fixed axis-key sequence, so there's no need to re-run JSON.stringify
// per image.
const AXISTAGS_ZYX = makeAxistags('zyx');
const AXISTAGS_YXC = makeAxistags('yxc');

/** `[y0:y1,x0:x1,c0:c1]`, matching ilastik's `serializerUtils.slicingToString`. */
function blockSliceString(y0, h, x0, w) {
  return `[${y0}:${y0 + h},${x0}:${x0 + w},0:1]`;
}

// Group-naming conventions from ilastik's own serializers (not just the
// format doc): dataSelectionSerializer.py ("lane{:04d}"),
// pixelClassificationSerializer.py ("labels{:03d}"), slotSerializer.py
// ("block{:04d}" — used here only for the single bounding-box block per lane).
const laneName = (i) => `lane${String(i).padStart(4, '0')}`;
const labelSetName = (i) => `labels${String(i).padStart(3, '0')}`;
const BLOCK_NAME = 'block0000';

/** '#rrggbb' -> [r,g,b]. Falls back to mid-gray for anything unparseable. */
function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
  if (!m) return [136, 136, 136];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

/**
 * Converts one image's sparse `{x,y,cls}` labels into a single dense block
 * covering their bounding box (0 = unlabeled, matching ilastik's sparse-block
 * convention; `cls + 1` elsewhere). Real ilastik further splits label data
 * into many small chunk blocks as an internal storage-cache optimization —
 * that's not a format requirement, so one bounding-box block per lane is a
 * valid, much simpler equivalent.
 *
 * Trade-off: the block is sized by the *span* between labels, not their
 * count, so two labels painted near opposite corners of a very large image
 * allocate a block covering the whole image. A pixel-count cap isn't a safe
 * fix here — it can't tell that case apart from a legitimately large, densely
 * labeled image — so this is an accepted simplicity trade-off, not a bug.
 * @returns {{y0:number,x0:number,h:number,w:number,data:Uint8Array}|null}
 *   null when there are no labels for this image.
 */
function labelsToBlock(labels) {
  if (!labels || labels.length === 0) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const { x, y } of labels) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const w = maxX - minX + 1, h = maxY - minY + 1;
  const data = new Uint8Array(w * h); // row-major (y, x)
  for (const { x, y, cls } of labels) data[(y - minY) * w + (x - minX)] = cls + 1;
  return { y0: minY, x0: minX, h, w, data };
}

/** Strips a trailing file extension, e.g. 'cells.tif' -> 'cells'. */
export function stripExtension(name) {
  return name.replace(/\.[^/.]+$/, '');
}

/**
 * Builds one lane's `Input Data/infos/laneNNNN/Raw Data` info group plus the
 * raw-pixel dataset it points at, embedded under `Input Data/local_data/`.
 *
 * Browsers never expose a real filesystem path for an uploaded file — only a
 * bare filename — so a `FileSystem`-location reference (as desktop ilastik
 * itself writes for on-disk images) would only work if the original file
 * happened to still be sitting next to the `.ilp`. Instead this mirrors
 * ilastik's own "copy into project" path: `ProjectInternalDatasetInfo`
 * (`opDataSelection.py`), which stores the array directly inside the project
 * at `Input Data/local_data/<datasetId>` and points the lane at it via
 * `location: "ProjectInternal"` + `inner_path`. That's the dispatch key
 * `dataSelectionSerializer.py._readDatasetInfo` actually switches on
 * (`__class__` when present, else `location`), so the exported `.ilp` is
 * fully self-contained — no separate image files to keep track of.
 * @returns {{infoGroup: Object, dataNode: Object}}
 */
function buildRawDataEntry(img) {
  const nickname = stripExtension(img.name);
  const datasetId = crypto.randomUUID();
  const innerPath = `/Input Data/local_data/${datasetId}`;

  const infoGroup = group('Raw Data', [
    dsStr('__class__', 'ProjectInternalDatasetInfo'),
    dsBoolScalar('allowLabels', true),
    dsStr('axistags', AXISTAGS_ZYX),
    dsStr('datasetId', datasetId),
    dsStr('display_mode', 'default'),
    dsStr('filePath', innerPath), // legacy-compat: real ilastik sets this to inner_path too
    dsStr('inner_path', innerPath),
    dsStr('location', 'ProjectInternal'),
    dsStr('nickname', nickname),
    dsBoolScalar('normalizeDisplay', false),
    dsInt64Array('shape', [1, img.height, img.width]),
  ]);

  // intensityArray is already Float32Array, laid out y*width+x (row-major,
  // x fastest) — exactly a [1, height, width] zyx array in C order, so no
  // reshaping is needed, just a direct byte copy.
  const dataNode = dsFloat32Nd(datasetId, [1, img.height, img.width], img.intensityArray);
  dataNode.attrs = [strAttr('axistags', AXISTAGS_ZYX)];

  return { infoGroup, dataNode };
}

/** Builds one lane's `PixelClassification/LabelSets/labelsNNN` group. */
function buildLabelSetGroup(img, index) {
  const block = labelsToBlock(img.labels);
  if (!block) return group(labelSetName(index), []);
  const dataset = dsUint8(BLOCK_NAME, [block.h, block.w, 1], block.data, [
    strAttr('axistags', AXISTAGS_YXC),
    strAttr('blockSlice', blockSliceString(block.y0, block.h, block.x0, block.w)),
  ]);
  return group(labelSetName(index), [dataset]);
}

// Standard ilastik feature bank metadata. This seeds FeatureSelections with a
// small starter selection (first two features x first two scales) the user
// refines in ilastik's own Feature Selection dialog on retrain — it has no
// relationship to Quantosaurus's own on-GPU feature bank (see NUM_FEATURES).
const FEATURE_IDS = [
  'GaussianSmoothing',
  'LaplacianOfGaussian',
  'GaussianGradientMagnitude',
  'DifferenceOfGaussians',
  'StructureTensorEigenvalues',
  'HessianOfGaussianEigenvalues',
];
const FEATURE_SCALES = [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0];

/**
 * Default `ParallelVigraRfLazyflowClassifierFactory` pickle (protocol-0, text
 * pickle — safe to build with plain string templating), parameterized by
 * tree count. Ilastik reads this as a hint for its own retrain; it does not
 * reference any trained weights.
 */
function buildClassifierFactoryPickle(numTrees) {
  const text = 'ccopy_reg\n_reconstructor\np0\n(clazyflow.classifiers.parallelVigraRfLazyflowClassifier\n' +
    'ParallelVigraRfLazyflowClassifierFactory\np1\nc__builtin__\nobject\np2\nNtp3\nRp4\n(dp5\n' +
    'VVERSION\np6\nL2L\nsV_num_trees\np7\n' + `L${numTrees}L` + '\nsV_label_proportion\np8\nNsV_variable_importance_path\n' +
    'p9\nNsV_variable_importance_enabled\np10\nI00\nsV_kwargs\np11\n(dp12\nsV_num_forests\np13\nL4L\nsb.';
  return textEncoder.encode(text);
}

/**
 * Builds a minimal, spec-compliant ilastik Pixel Classification `.ilp`
 * project from the current app state: one lane per loaded image, the user's
 * painted labels as one bounding-box block per image, label names/colors,
 * and a starter feature selection. No trained classifier is embedded (see
 * module doc above).
 * @param {Object} state - Shared app state (state.images, state.rf,
 *   state.labelColors).
 * @param {Object} [options]
 * @param {Array<string>} [options.classNames] - Per-class display names;
 *   defaults to "Class 1", "Class 2", ....
 * @param {string} [options.time] - Overrides the project's saved timestamp
 *   (mainly for deterministic tests); defaults to the current time.
 * @returns {Uint8Array} The complete `.ilp` (HDF5) file bytes.
 */
export function buildIlpProject(state, options = {}) {
  const images = state.images ?? [];
  const numClasses = state.rf?.numClasses ?? RF_CONFIG.numClasses;
  const numTrees = state.rf?.numTrees ?? RF_CONFIG.numTrees;
  const classNames = options.classNames ?? Array.from({ length: numClasses }, (_, i) => `Class ${i + 1}`);
  const colorRows = Array.from({ length: numClasses }, (_, i) => hexToRgb(state.labelColors?.[i]));

  const rawDataEntries = images.map(buildRawDataEntry);
  const inputData = group('Input Data', [
    dsStr('StorageVersion', '0.2'),
    dsStrArray('Role Names', ['Raw Data', 'Prediction Mask']),
    group('infos', rawDataEntries.map((e, i) => group(laneName(i), [e.infoGroup]))),
    group('local_data', rawDataEntries.map((e) => e.dataNode)),
  ]);

  const selectionMatrix = FEATURE_IDS.map((_, fi) => FEATURE_SCALES.map((_, si) => fi < 2 && si < 2));
  const featureSelections = group('FeatureSelections', [
    dsStr('StorageVersion', '0.1'),
    dsStrArray('FeatureIds', FEATURE_IDS),
    dsFloat64Array('Scales', FEATURE_SCALES),
    dsBoolMatrix('SelectionMatrix', selectionMatrix),
    dsBoolArray('ComputeIn2d', FEATURE_SCALES.map(() => true)),
  ]);

  const pixelClassification = group('PixelClassification', [
    dsStr('StorageVersion', '0.1'),
    // The HDF5 field name must stay the literal 'LabelNames' — that's the
    // key ilastik's own PixelClassificationSerializer reads/writes
    // (SerialListSlot(operator.LabelNames)); classNames here is just our
    // local variable name, unrelated to the on-disk schema.
    dsStrArray('LabelNames', classNames),
    dsInt32Matrix('LabelColors', colorRows),
    dsInt32Matrix('PmapColors', colorRows),
    group('Bookmarks', []),
    group('LabelSets', images.map((img, i) => buildLabelSetGroup(img, i))),
    dsBytes('ClassifierFactory', buildClassifierFactoryPickle(numTrees)),
  ]);

  const root = [
    dsStr('ilastikVersion', '1.4.0'),
    dsStr('workflowName', 'Pixel Classification'),
    dsStr('time', options.time ?? formatCtime(new Date())),
    dsInt64Scalar('currentApplet', 0),
    inputData,
    featureSelections,
    pixelClassification,
  ];

  return serializeHdf5(root);
}
