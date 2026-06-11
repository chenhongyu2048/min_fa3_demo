import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

this_dir = Path(__file__).resolve().parent
repo_root = this_dir

# Resolve the CUTLASS include directory from either CUTLASS_DIR or the vendored third_party copy.
def resolve_cutlass_include_dir() -> Path:
    env_value = os.environ.get("CUTLASS_DIR")
    if env_value:
        cutlass_path = Path(env_value).expanduser().resolve()
        include_dir = cutlass_path / "include"
        if include_dir.is_dir():
            return include_dir
        if cutlass_path.name == "include" and cutlass_path.is_dir():
            return cutlass_path
        raise RuntimeError(
            "CUTLASS_DIR must point to a CUTLASS root containing include/ or directly to the include/ directory. "
            f"Got: {cutlass_path}"
        )

    default_include_dir = repo_root / "third_party" / "cutlass" / "include"
    if default_include_dir.is_dir():
        return default_include_dir

    raise RuntimeError(
        "Could not find CUTLASS headers. Set CUTLASS_DIR=/path/to/cutlass "
        "or CUTLASS_DIR=/path/to/cutlass/include before building."
    )

cutlass_include_dir = resolve_cutlass_include_dir()
print(f"Using CUTLASS headers from: {cutlass_include_dir}")

# Find a linkable libcuda location for builds that directly depend on the CUDA Driver API.
def resolve_cuda_link_dir() -> tuple[Path, bool]:
    search_dirs: list[Path] = []
    for env_name in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        env_value = os.environ.get(env_name)
        if not env_value:
            continue
        search_dirs.extend(Path(entry).expanduser() for entry in env_value.split(":") if entry)

    search_dirs.extend(
        Path(path)
        for path in (
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/wsl/lib",
        )
    )

    seen_dirs: set[Path] = set()
    for candidate_dir in search_dirs:
        resolved_dir = candidate_dir.resolve()
        if resolved_dir in seen_dirs:
            continue
        seen_dirs.add(resolved_dir)
        if (resolved_dir / "libcuda.so").is_file():
            return resolved_dir, resolved_dir.name == "stubs"

    if CUDA_HOME:
        for relative_stub_dir in (Path("lib64") / "stubs", Path("lib") / "stubs"):
            candidate_dir = Path(CUDA_HOME).resolve() / relative_stub_dir
            if (candidate_dir / "libcuda.so").is_file():
                return candidate_dir, True

    raise RuntimeError(
        "Could not find a linkable libcuda.so. On this Slurm/module setup, "
        "either make the driver development library visible in LIBRARY_PATH/LD_LIBRARY_PATH "
        "or provide CUDA_HOME with toolkit stubs under lib64/stubs."
    )

cuda_link_dir, using_cuda_stubs = resolve_cuda_link_dir()
if using_cuda_stubs:
    print(
        "Linking libcuda from CUDA toolkit stubs: "
        f"{cuda_link_dir} (build-time only; runtime still requires a real driver on the compute node)"
    )
else:
    print(f"Linking libcuda from driver library directory: {cuda_link_dir}")

# Build the single PyTorch CUDA extension with the local sources, headers, and SM90 compile flags.
ext_modules = [
    CUDAExtension(
        name="_min_fa3_op",
        sources=[
            "bindings.cpp",
            "csrc/min_fa3_launch.cu",
            "csrc/min_fa3_kernel.cu",
            "csrc/min_fa3_varlen_prepare_scheduler.cu",
            "csrc/min_fa3_varlen_launch.cu",
            "csrc/min_fa3_varlen_kernel.cu",
            "csrc/min_fa3_varlen_ring_launch.cu",
            "csrc/min_fa3_varlen_ring_bindings.cu",
            "csrc/parallel/remote_load.cu",
            "csrc/parallel/remote_load_bindings.cu",
        ],
        include_dirs=[
            str(this_dir / "include"),
            str(this_dir / "include" / "hopper_compat"),
            str(this_dir / "third_party" / "ThunderKittens" / "include"),
            str(cutlass_include_dir),
        ],
        library_dirs=[str(cuda_link_dir)],
        libraries=["cuda"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++20", "-DKITTENS_SM90"],
            "nvcc": [
                "-O3",
                "-std=c++20",
                "--use_fast_math",
                "-lineinfo",
                "-diag-suppress=3189",
                "--expt-extended-lambda",
                "-DKITTENS_SM90",
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
                "-DCUTLASS_ENABLE_GDC_FOR_SM90",
                "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                "-DNDEBUG",
                "-gencode",
                "arch=compute_90a,code=sm_90a",
            ],
        },
    )
]

setup(
    name="min_fa3_demo",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    packages=[],
    py_modules=["min_fa3_op"],
)
