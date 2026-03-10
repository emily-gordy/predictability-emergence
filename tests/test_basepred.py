import subprocess

def run(cmd):
    tokens = cmd.split()
    return subprocess.run(
                          tokens, 
                          capture_output=True,  # <-- THIS
                          text=True             # <-- decode bytes to str)
    )

def test_help():
    result = run("basepred --help")
    assert result.returncode == 0

def test_small(data_dir):
    if data_dir is not None:
        result = run(f"basepred --data_dir {data_dir} --outputavgtime=5 --ssps 126 --baseline_era 1900 1910 --experiment_era 2000 2010 --test 38 49")
        assert result.returncode == 0
