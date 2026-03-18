"""
Proto Compilation Script
Generates Python gRPC code from .proto files
"""
import subprocess
import sys
import os

def compile_proto():
    """Compile robot_interface.proto to Python code."""
    proto_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = proto_dir

    proto_file = os.path.join(proto_dir, "robot_interface.proto")

    if not os.path.exists(proto_file):
        print(f"Error: {proto_file} not found")
        return False

    # Check if grpcio-tools is installed
    try:
        import grpc_tools.protoc
    except ImportError:
        print("Error: grpcio-tools not installed")
        print("Install with: pip install grpcio-tools")
        return False

    # Compile proto
    print(f"Compiling {proto_file}...")

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        proto_file
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        return False

    print("Compilation successful!")
    print(f"Generated files:")
    print(f"  - {output_dir}/robot_interface_pb2.py")
    print(f"  - {output_dir}/robot_interface_pb2_grpc.py")

    return True


if __name__ == "__main__":
    success = compile_proto()
    sys.exit(0 if success else 1)
