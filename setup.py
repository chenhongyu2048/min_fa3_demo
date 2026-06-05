import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).resolve().parent
repo_root = this_dir

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

ext_modules = [
    CUDAExtension(
        name="_min_fa3_op",
        sources=[
            str(this_dir / "bindings.cpp"),
            str(this_dir / "csrc" / "min_fa3_launch.cu"),
            str(this_dir / "csrc" / "min_fa3_kernel.cu"),
            str(this_dir / "csrc" / "min_fa3_varlen_prepare_scheduler.cu"),
            str(this_dir / "csrc" / "min_fa3_varlen_launch.cu"),
            str(this_dir / "csrc" / "min_fa3_varlen_kernel.cu"),
        ],
        include_dirs=[
            str(this_dir / "include"),
            str(this_dir / "include" / "hopper_compat"),
            str(cutlass_include_dir),
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-lineinfo",
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
)
