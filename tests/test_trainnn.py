import subprocess

def run(cmd):
    tokens = cmd.split()
    return subprocess.run(tokens)

def test_help():
    result = run("trainnn --help")

