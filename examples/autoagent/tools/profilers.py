import subprocess
import sys
from pathlib import Path
from typing import Optional

from tools.capabilities import detect_runtime_capabilities
from tools.kernel_pipeline import write_profile_script


class ProfileExecutor:
    name = "unknown"

    def describe_unavailability(self) -> Optional[str]:
        return None

    def profile(
        self,
        output_so: Path,
        meta_path: Path,
        report_path: Path,
        kernel_config_path: Path,
    ) -> bool:
        raise NotImplementedError


class DisabledProfileExecutor(ProfileExecutor):
    name = "disabled"

    def describe_unavailability(self) -> Optional[str]:
        return (
            "profiling is disabled for this runtime. OpenShell standard sandboxes block "
            "ptrace and perf_event_open, so NCU should be run in a dedicated profiling environment."
        )

    def profile(self, output_so, meta_path, report_path, kernel_config_path) -> bool:
        raise RuntimeError(self.describe_unavailability())


class NcuProfileExecutor(ProfileExecutor):
    name = "ncu"

    def __init__(self, settings):
        self.settings = settings

    def describe_unavailability(self) -> Optional[str]:
        status = detect_runtime_capabilities(self.settings).get("ncu")
        if not status.available:
            return f"the 'ncu' runtime capability is unavailable: {status.detail}"
        return None

    def profile(self, output_so, meta_path, report_path, kernel_config_path) -> bool:
        reason = self.describe_unavailability()
        if reason:
            raise RuntimeError(reason)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        script_path = report_path.with_suffix(report_path.suffix + ".profile_tmp.py")
        kernel_name = write_profile_script(
            output_so=output_so,
            meta_path=meta_path,
            config_path=kernel_config_path,
            workspace_root=self.settings.workspace,
            script_path=script_path,
        )
        cmd = [
            *self.settings.ncu_command,
            "--set",
            ",".join(self.settings.ncu_sets),
            "--kernel-name",
            kernel_name,
            "--launch-skip",
            "0",
            "--launch-count",
            "1",
            "--log-file",
            str(report_path),
            "--force-overwrite",
            *(self.settings.python_command or (sys.executable,)),
            str(script_path),
        ]

        print(f"Profiling: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"Profiling failed with exit code {result.returncode}")
            print(f"Profile script: {script_path}")
            if result.stdout:
                print(f"NCU STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"NCU STDERR:\n{result.stderr}")
            if report_path.exists():
                log_text = report_path.read_text(encoding="utf-8", errors="replace")
                if log_text:
                    print(f"NCU LOG FILE ({report_path}):\n{log_text}")
            return False
        return True


def create_profile_executor(name: str, settings) -> ProfileExecutor:
    normalized = name.strip().lower()
    if normalized == "disabled":
        return DisabledProfileExecutor()
    if normalized == "ncu":
        return NcuProfileExecutor(settings)
    raise RuntimeError(f"Unsupported profile executor: {name}")
