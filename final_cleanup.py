import os
import subprocess

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

def run_git_command(command, shell=False):
    if shell:
        subprocess.run(command, shell=True, cwd=WORKING_DIR, check=True)
    else:
        subprocess.run(command.split(), cwd=WORKING_DIR, check=True)

def main():
    # Remove all .vscode folders
    for root, dirs, files in os.walk(WORKING_DIR):
        if '.vscode' in dirs:
            vscode_path = os.path.join(root, '.vscode')
            subprocess.run(['rm', '-rf', vscode_path], check=True)

    # Update .gitignore files
    gitignore_paths = [
        "Democratizing-AI-Lang/.gitignore",
        "Domocratizing-AI-OS/.gitignore"
    ]
    
    for gitignore_path in gitignore_paths:
        full_path = os.path.join(WORKING_DIR, gitignore_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = f.read()
            
            if '.vscode/' not in content:
                with open(full_path, 'a') as f:
                    f.write('\n# VSCode settings\n.vscode/\n')

    # Remove cleanup scripts and commit changes
    cleanup_files = [
        'cleanup.py',
        'fix_launch_json.py',
        'final_cleanup.py'
    ]
    
    # Add and commit changes
    run_git_command('git add -A')
    run_git_command('git commit -m "Remove .vscode folders and update .gitignore"')

    # Remove cleanup scripts
    for file in cleanup_files:
        if os.path.exists(file):
            os.remove(file)

if __name__ == "__main__":
    main()
