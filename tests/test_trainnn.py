import subprocess

def run(cmd):
    tokens = cmd.split()
    return subprocess.run(tokens)

def test_help():
    result = run("trainnn --help")
    assert result.returncode == 0

def test_small(data_dir):
    if data_dir is not None:
        result = run(f"trainnn --data-dir {data_dir} --epochs 3 --ssps 126 245")

