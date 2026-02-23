import subprocess
import re

best_validation_accuracy_pat = re.compile(r'Best validation accuracy: ([\d\.]+)')

def get_best_validation_accuracy(txt):
    m = re.search(best_validation_accuracy_pat, txt)
    if m:
        return float(m.group(1))
    return None


def run(cmd):
    tokens = cmd.split()
    return subprocess.run(
                          tokens, 
                          capture_output=True,  # <-- THIS
                          text=True             # <-- decode bytes to str)
    )

def test_help():
    result = run("trainnn --help")
    assert result.returncode == 0

def test_small(data_dir):
    if data_dir is not None:
        result = run(f"trainnn --data_dir {data_dir} --epochs 3 --ssps 126 245 --seed 1234")
        assert result.returncode == 0
        best_validation_accuracy = get_best_validation_accuracy(result.stderr)
        assert abs(best_validation_accuracy - 0.752257) < 1.e-4
