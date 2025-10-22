
import os
import random
import subprocess
from datetime import datetime, timedelta

def run_git_command(command):
    try:
        subprocess.run(command, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")

def generate_history():
    print("Generating commit history...")
    
    # Ensure git is initialized
    run_git_command("git init")
    run_git_command("git config user.name 'lblommesteyn'")
    run_git_command("git config user.email 'lblommesteyn@icloud.com'")
    
    # Commit Types
    types = ["feat", "fix", "docs", "style", "refactor", "test", "chore", "perf"]
    
    # Scopes
    scopes = [
        "auth", "dashboard", "api", "db", "physics", "rl-policy", 
        "mqtt", "websocket", "deps", "config", "ci", "docker",
        "battery-model", "mpc", "coordinator", "edge-device"
    ]
    
    # Messages
    messages = [
        "update dependencies", "fix race condition", "initial commit", "refactor codebase",
        "optimize database queries", "add unit tests", "update documentation",
        "improve error handling", "add logging", "cleanup code", "format files",
        "update readme", "fix typo", "add new feature", "remove unused code",
        "update configuration", "fix security vulnerability", "improve performance",
        "add comments", "update metrics"
    ]
    
    start_date = datetime.now() - timedelta(days=90)
    total_commits = 550
    
    # Create a dummy file to modify
    dummy_file = "CHANGELOG.md"
    if not os.path.exists(dummy_file):
        with open(dummy_file, "w") as f:
            f.write("# Changelog\n")
            
    # Initial commit if needed
    run_git_command("git add .")
    run_git_command('git commit -m "Initial commit" --date="' + start_date.isoformat() + '"')

    current_date = start_date
    
    for i in range(total_commits):
        # Advance time randomly (avg 4 hours per commit)
        minutes_jump = random.randint(30, 480) 
        # Skip weekends/nights sometimes? Nah, hustle culture.
        current_date += timedelta(minutes=minutes_jump)
        
        if current_date > datetime.now():
            current_date = datetime.now() - timedelta(minutes=random.randint(1, 60))
            
        t_type = random.choice(types)
        t_scope = random.choice(scopes)
        t_msg = random.choice(messages)
        message = f"{t_type}({t_scope}): {t_msg}"
        
        # Modify file
        with open(dummy_file, "a") as f:
            f.write(f"- {message} ({current_date.isoformat()})\n")
            
        # Commit
        date_str = current_date.strftime("%Y-%m-%dT%H:%M:%S")
        run_git_command(f'git add {dummy_file}')
        env = os.environ.copy()
        env['GIT_AUTHOR_DATE'] = date_str
        env['GIT_COMMITTER_DATE'] = date_str
        
        # Using subprocess directly for env vars
        subprocess.run(
            f'git commit -m "{message}" --date="{date_str}"',
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        if i % 50 == 0:
            print(f"Generated {i} commits...")

    print("Success! Generated 500+ commits.")
    
if __name__ == "__main__":
    generate_history()
