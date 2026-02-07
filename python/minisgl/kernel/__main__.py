
assert __name__ == "__main__"


def generate_clangd():
    """
    生成 .clangd 配置文件
    
    自动检测系统 CUDA 版本和头文件路径，生成 .clangd 文件，
    以便在使用 VSCode + clangd 开发 C++/CUDA 代码时提供智能提示。
    """
    import os
    import subprocess

    from minisgl.kernel.utils import DEFAULT_INCLUDE
    from minisgl.utils import init_logger
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    logger = init_logger(__name__)
    logger.info("Generating .clangd file...")
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    
    # 获取当前 GPU 的 Compute Capability
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
    )
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    major, minor = compute_cap.split(".")
    
    # 构造编译参数
    compile_flags = ",\n    ".join(
        [
            "-xcuda",
            f"--cuda-gpu-arch=sm_{major}{minor}",
            "-std=c++20",
            "-Wall",
            "-Wextra",
        ]
        + [f"-isystem{path}" for path in include_paths]
    )
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    if os.path.exists(".clangd"):
        logger.warning(".clangd file already exists, nothing done.")
        logger.warning(f"suggested content: {clangd_content}")
    else:
        with open(".clangd", "w") as f:
            f.write(clangd_content)
        logger.info(".clangd file generated.")


generate_clangd()
