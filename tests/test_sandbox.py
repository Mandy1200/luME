import unittest
from lume.sandbox import verify_code_safety

class TestSandboxSafety(unittest.TestCase):
    def test_allowed_code(self):
        code = """
x = 10
y = 20
result = x + y
print(f"Result is {result}")
"""
        is_safe, errors = verify_code_safety(code, [])
        self.assertTrue(is_safe)
        self.assertEqual(len(errors), 0)

    def test_blocked_import_os(self):
        code = """
import os
os.system("ls")
"""
        is_safe, errors = verify_code_safety(code, [])
        self.assertFalse(is_safe)
        self.assertTrue(any("forbidden module 'os'" in err for err in errors))

    def test_blocked_import_subprocess(self):
        code = """
from subprocess import run
run(["ls"])
"""
        is_safe, errors = verify_code_safety(code, [])
        self.assertFalse(is_safe)
        self.assertTrue(any("forbidden module 'subprocess'" in err for err in errors))

    def test_blocked_builtins(self):
        code = """
eval("1 + 1")
"""
        is_safe, errors = verify_code_safety(code, [])
        self.assertFalse(is_safe)
        self.assertTrue(any("unsafe built-in function 'eval'" in err for err in errors))

    def test_blocked_open_call(self):
        code = """
with open("test.txt", "w") as f:
    f.write("hello")
"""
        is_safe, errors = verify_code_safety(code, [])
        self.assertFalse(is_safe)
        self.assertTrue(any("unsafe built-in function 'open'" in err for err in errors))

    def test_blocked_magic_method_access(self):
        code = """
class MyClass:
    pass
# Attempt to access classes/subclasses bypasses
MyClass.__subclasses__()
"""
        is_safe, errors = verify_code_safety(code, [])
        self.assertFalse(is_safe)
        self.assertTrue(any("sensitive attribute '__subclasses__'" in err for err in errors))

if __name__ == '__main__':
    unittest.main()
