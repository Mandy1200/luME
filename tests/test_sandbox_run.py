import unittest
import shutil
import os
from lume.sandbox import run_code_in_sandbox

class TestSandboxRun(unittest.TestCase):
    def setUp(self):
        self.sandbox_dir = "test_sandbox_workspace"
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)

    def test_run_write_read_list(self):
        # We write code that uses the safe injected helpers read_file and write_file.
        # This code is AST-safe because it doesn't call open() or import os.
        code = """
write_file("hello.txt", "Hello Sandbox World!")
content = read_file("hello.txt")
files = list_files()
print(f"Content: {content}")
print(f"Files: {files}")
"""
        exit_code, stdout, stderr = run_code_in_sandbox(code, self.sandbox_dir)
        self.assertEqual(exit_code, 0)
        self.assertIn("Content: Hello Sandbox World!", stdout)
        self.assertIn("'hello.txt'", stdout)
        
        # Verify it actually wrote the file into our sandbox directory
        file_on_disk = os.path.join(self.sandbox_dir, "hello.txt")
        self.assertTrue(os.path.exists(file_on_disk))
        with open(file_on_disk, 'r') as f:
            self.assertEqual(f.read(), "Hello Sandbox World!")

    def test_run_path_traversal_blocked(self):
        # Code attempts directory traversal via injected helper
        code = """
try:
    write_file("../forbidden.txt", "dangerous content")
except PermissionError as e:
    print(f"Trapped traversal: {e}")
"""
        exit_code, stdout, stderr = run_code_in_sandbox(code, self.sandbox_dir)
        self.assertEqual(exit_code, 0)
        self.assertIn("Trapped traversal: Access denied: path is outside the workspace sandbox.", stdout)
        self.assertFalse(os.path.exists(os.path.join(self.sandbox_dir, "../forbidden.txt")))

if __name__ == '__main__':
    unittest.main()
