import subprocess
from typing import Optional


def bash(command: str, cwd: Optional[str] = None) -> str:
    """
    Execute a bash command and return the results.
    Useful for finding files in the codebase and reading them.
    ALWAYS use this tool to find files irrespective of the task at hand.

    Args:
        command: The full bash command to execute
                 Example: "rg 'def main' -t py" or "ls -la" or "cat file.txt"
        cwd: Working directory to execute the command in (optional)
             If None, uses the current directory

    Returns:
        Command output as a string

    Example:
        >>> bash("rg 'def main' -t py")
        src/main.py:10:def main():
        src/utils.py:5:def main_helper():

        >>> bash("ls -la", cwd="/path/to/repo")
        >>> bash("rg --files -t py", cwd="./src")
        >>> bash("cat README.md")  # Read file contents

    Note:
        Common commands:
        - rg: ripgrep for searching code
        - ls: list directory contents
        - cat: read file contents
        - find: find files by name
        - grep: search in files
    """

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,  # 30 second timeout
            cwd=cwd,  # Set working directory
        )

        # Combine stdout and stderr
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            # Command succeeded
            if not output:
                return "Command executed successfully (no output)."

            # Truncate to 200 lines and append message with remaining lines and total lines if more than 200
            if len(output) > 200:
                return f"Output truncated to 200 lines. Total lines: {len(output.splitlines())}\n\nOutput:\n{output}"

            return output
        else:
            # Command failed
            if error:
                return f"Error (exit code {result.returncode}):\n{error}"
            elif output:
                return f"Exit code {result.returncode}:\n{output}"
            else:
                return f"Command failed with exit code {result.returncode}"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error: {str(e)}"
