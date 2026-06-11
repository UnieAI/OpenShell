import os
import subprocess

class NCUProfiler:
    def __init__(self, report_dir="ncu_reports"):
        self.report_dir = report_dir
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def profile(self, target_command, report_name, kernel_name=None, sets=None, metrics=None, sections=None):
        """
        Profiles a command using NVIDIA Nsight Compute (ncu).
        """
        report_path = os.path.join(self.report_dir, report_name)
        cmd = ["ncu", "--force-overwrite", "-o", report_path]
        
        if kernel_name:
            cmd += ["-k", kernel_name]
        if sets:
            cmd += ["--set", ",".join(sets)]
        if metrics:
            cmd += ["--metrics", ",".join(metrics)]
        if sections:
            for section in sections:
                cmd += ["--section", section]
        
        cmd += target_command
        
        print(f"Profiling with NCU: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Profiling report saved to: {report_path}.ncu-rep")
