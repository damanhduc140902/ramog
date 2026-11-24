"""Exact-size page-locked host memory for the expert blobs (the cold store the GPU reads zero-copy).

torch's pinned allocator (CachingHostAllocator, behind Tensor.pin_memory()) rounds every request up to a power of
two: a 182 MiB Qwen3-30B cold layer takes 256 MiB of locked RAM (12 GiB instead of 8.5 GiB for the store), a
0.76 GB Qwen3-235B layer 1 GiB. Here the bytes go into an ordinary page-aligned buffer that is then page-locked in
place with cudaHostRegister (hipHostRegister on ROCm; both through torch.cuda.cudart()), flags Portable | Mapped:
exact size and no second copy. Kernels read it zero-copy (Triton translates the host pointer into its device
alias, which may differ from the host address, e.g. on Windows/WDDM), and torch reports the tensor as pinned, so
copies from it stay asynchronous and .pin_memory() on it is a no-op.

Every registration is undone after a device synchronize by release() (call it at teardown, once nothing reads the
blobs any more), when its buffer is garbage collected, or at interpreter exit. If registering is unavailable or
fails (or ENGINE_EXACT_PIN=0), the functions fall back to Tensor.pin_memory() (power-of-two rounded, as before).
A failed runtime call also leaves the runtime's per-thread "last error" set, which torch's next kernel-launch check
would raise as an unrelated "invalid argument"; it is read and cleared right away (_clear_last_error).
"""

from __future__ import annotations

import atexit
import ctypes
import os
import threading
import weakref

import numpy as np
import torch

PAGE = 4096
_FLAGS = 0x01 | 0x02  # cudaHostRegisterPortable | cudaHostRegisterMapped (hipHostRegister uses the same values)
_lock = threading.Lock()
_live: dict[int, int] = {}  # registered host address -> bytes
_state = {"disabled": os.environ.get("ENGINE_EXACT_PIN", "1") == "0", "warned": False, "fallback_bytes": 0}


def _rt():
    return torch.cuda.cudart()


def _round_up(n: int, a: int) -> int:
    return -(-n // a) * a


def _runtime_libs() -> list:
    """The CUDA / HIP runtime libraries already loaded in this process (torch's; the per-thread last error lives in
    the runtime instance that made the failing call). Only loaded libraries are looked up, none is loaded."""
    prefix = ("amdhip64", "libamdhip64") if torch.version.hip else ("cudart", "libcudart")
    libs = []
    try:
        if os.name == "nt":
            import ctypes.wintypes as wt

            k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wt.HANDLE
            psapi.EnumProcessModules.argtypes = [wt.HANDLE, ctypes.POINTER(wt.HMODULE), wt.DWORD,
                                                 ctypes.POINTER(wt.DWORD)]
            psapi.GetModuleBaseNameW.argtypes = [wt.HANDLE, wt.HMODULE, wt.LPWSTR, wt.DWORD]
            proc = k32.GetCurrentProcess()
            mods = (wt.HMODULE * 4096)()
            need = wt.DWORD()
            if psapi.EnumProcessModules(proc, mods, ctypes.sizeof(mods), ctypes.byref(need)):
                name = ctypes.create_unicode_buffer(260)
                for h in mods[: need.value // ctypes.sizeof(wt.HMODULE)]:
                    if h and psapi.GetModuleBaseNameW(proc, h, name, 260) and name.value.lower().startswith(prefix):
                        libs.append(ctypes.WinDLL(name.value, handle=h))
        else:
            with open("/proc/self/maps") as f:
                paths = {ln.split(maxsplit=5)[-1].strip() for ln in f if "/" in ln}
            for path in sorted(paths):
                if os.path.basename(path).startswith(prefix[1]) and os.path.exists(path):
                    try:
                        libs.append(ctypes.CDLL(path))  # already loaded: the same instance, nothing new is mapped
                    except OSError:
                        pass
    except Exception:
        pass
    return libs


def _clear_last_error() -> str:
    """Read and reset the runtime's last error after a failed runtime call (cudaGetLastError / hipGetLastError of
    every loaded runtime library; torch.cuda.cudart() has no binding for it), then launch one tiny kernel: torch's
    launch check reads and resets the error too, so this also covers a runtime linked statically into torch.
    Returns how it went, for the log and the tests."""
    fn = "hipGetLastError" if torch.version.hip else "cudaGetLastError"
    done = 0
    for lib in _runtime_libs():
        f = getattr(lib, fn, None)
        if f is not None:
            f.restype = ctypes.c_int
            f()
            done += 1
    how = f"{fn} x{done}" if done else f"no loaded library exports {fn}"
    try:  # fill with 1, not 0: a zero fill may become a memset, whose check does not read the last error
        torch.empty(1, device=torch.device("cuda", torch.cuda.current_device())).fill_(1)
        return how
    except Exception:
        return how + "; a probe launch consumed it"


def _unregister(addr: int) -> None:
    try:
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            return  # never synchronize inside a graph capture: release() / exit unregisters it later
    except Exception:
        pass
    with _lock:
        if _live.pop(addr, None) is None:
            return
    try:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()  # no kernel or copy may still read the buffer
        rt = _rt()
        if rt.cudaHostUnregister(addr) != rt.cudaError.success:
            _clear_last_error()  # or torch's next launch check would raise it
    except Exception:  # interpreter shutdown / context already gone: the driver unpins at process exit anyway
        pass


def _register(addr: int, nbytes: int) -> bool:
    if _state["disabled"]:
        return False
    try:
        rt = _rt()
        res = rt.cudaHostRegister(addr, nbytes, _FLAGS)
        if res != rt.cudaError.success:
            how = _clear_last_error()  # the fallback must not leave a pending error for the next launch
            raise RuntimeError(f"cudaHostRegister({nbytes} B): {rt.cudaGetErrorString(res)}; error cleared by {how}")
    except Exception as e:
        if not _state["warned"]:
            print(f"behemoe.hostmem: exact-size pinning unavailable ({e}); falling back to pin_memory()", flush=True)
            _state["warned"] = True
        return False
    with _lock:
        _live[addr] = nbytes
    return True


def _aligned(nbytes: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(owner buffer, page-aligned view of nbytes, registrable length = whole pages inside the owner)."""
    reg = _round_up(max(nbytes, 1), PAGE)
    buf = np.empty(reg + PAGE, dtype=np.uint8)
    off = (-buf.ctypes.data) % PAGE
    return buf, buf[off:off + nbytes], reg


def _lock_in_place(buf: np.ndarray, view: np.ndarray, reg: int) -> torch.Tensor | None:
    addr = view.ctypes.data
    if not _register(addr, reg):
        return None
    weakref.finalize(buf, _unregister, addr)  # runs before numpy frees the memory
    return torch.from_numpy(view)  # keeps view (and through it buf) alive


def read_file(path: str, nbytes: int | None = None) -> torch.Tensor:
    """The bytes of `path` as a page-locked uint8 CPU tensor of exactly the file size (`nbytes`, if given, must
    match it). Read straight into the buffer that is then locked: no pageable copy, no rounding."""
    size = os.path.getsize(path)
    if nbytes is not None and size != nbytes:
        raise ValueError(f"{path}: {size} bytes, expected {nbytes}")
    buf, view, reg = _aligned(size)
    mv = memoryview(view)
    with open(path, "rb", buffering=0) as f:
        got = 0
        while got < size:
            n = f.readinto(mv[got:])
            if not n:
                raise EOFError(f"{path}: read {got} of {size} bytes")
            got += n
    del mv
    t = _lock_in_place(buf, view, reg)
    if t is None:
        _state["fallback_bytes"] += size
        return torch.from_numpy(view).pin_memory()
    return t


def read_rows(path: str, nbytes: int, rows) -> torch.Tensor:
    """Rows `rows` (in that order) of the uint8 [E, nbytes] file `path` (row e = expert e: core's exact-size layer
    file) as ONE page-locked uint8 CPU tensor [len(rows), nbytes] of exactly that size, read straight into the buffer
    that is then locked, one seek + read per run of consecutive rows (design-quality WP12: the partial host load of a
    split). No rows: an empty, unpinned tensor."""
    rows = [int(r) for r in rows]
    size = os.path.getsize(path)
    if nbytes <= 0 or size % nbytes:
        raise ValueError(f"{path}: {size} bytes is not a multiple of {nbytes}")
    bad = [r for r in rows if not 0 <= r < size // nbytes]
    if bad:
        raise IndexError(f"{path}: rows {bad[:4]} outside 0..{size // nbytes - 1}")
    total = len(rows) * nbytes
    if total == 0:
        return torch.empty((0, nbytes), dtype=torch.uint8)
    buf, view, reg = _aligned(total)
    mv = memoryview(view)
    with open(path, "rb", buffering=0) as f:
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and rows[j] == rows[j - 1] + 1:
                j += 1
            n = (j - i) * nbytes
            dst = mv[i * nbytes:i * nbytes + n]
            f.seek(rows[i] * nbytes)
            got = 0
            while got < n:
                k = f.readinto(dst[got:])
                if not k:
                    raise EOFError(f"{path}: read {got} of {n} bytes at row {rows[i]}")
                got += k
            i = j
    del mv, dst
    t = _lock_in_place(buf, view, reg)
    if t is None:
        _state["fallback_bytes"] += total
        t = torch.from_numpy(view).pin_memory()
    return t.view(len(rows), nbytes)


def pin_copy(t: torch.Tensor) -> torch.Tensor:
    """Page-locked copy of a CPU tensor at exact size (like t.pin_memory() without the power-of-two rounding)."""
    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    if nbytes == 0:
        return t.pin_memory()
    buf, view, reg = _aligned(nbytes)
    out = _lock_in_place(buf, view, reg)
    if out is None:
        _state["fallback_bytes"] += nbytes
        return t.pin_memory()
    out = out.view(t.dtype).view(t.shape)
    out.copy_(t)
    return out


def registered_bytes() -> int:
    with _lock:
        return sum(_live.values())


def fallback_bytes() -> int:
    """Bytes that went through pin_memory() because exact-size registration failed."""
    return _state["fallback_bytes"]


def peak_rss() -> int:
    """Peak resident memory of this process in bytes (Windows: peak working set; elsewhere ru_maxrss)."""
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t)] + [(f"f{i}", ctypes.c_size_t) for i in range(6)]

        c = PMC()
        c.cb = ctypes.sizeof(c)
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return int(c.PeakWorkingSetSize)
    import resource
    import sys

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def release() -> int:
    """Unregister every buffer still registered (teardown: nothing may read the blobs afterwards). The tensors stay
    valid as ordinary pageable memory. Returns the bytes released."""
    with _lock:
        addrs = list(_live)
    total = registered_bytes()
    for a in addrs:
        _unregister(a)
    return total


atexit.register(release)
