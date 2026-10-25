"""Ensure compact prefixes cover argument reads in both actual backends.

This couples the format audit to consumer source, catching a later backend change
that starts reading another argument without updating the compact representation.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parent.parent


class CommandRecordAccessTests(unittest.TestCase):
    def test_compact_prefix_covers_backend_reads(self):
        header = (ROOT / 'runtime/nvn/command_record.h').read_text()
        function = header.split('nvn_cmd_compact_header', 1)[1].split('static inline', 1)[0]
        limits = {}
        pending = []
        for token in re.finditer(r'case (CMD_\w+):|return (\d+);', function):
            if token[1]:
                pending.append(token[1])
            else:
                for op in pending:
                    limits[op] = int(token[2])
                pending = []
        self.assertGreater(len(limits), 20)
        checked = set()
        for name in ['runtime/nvn/vulkan/exec.c', 'runtime/nvn/renderer.m']:
            source = (ROOT / name).read_text()
            # Definitions only (not the earlier recursive calls/prototypes).
            executor = source.split('static void exec_list(uint64_t handle, int depth) {', 1)[1]
            executor = executor.split('\nvoid backend_execute', 1)[0]
            cases = list(re.finditer(r'case (CMD_\w+):', executor))
            for i, case in enumerate(cases):
                op = case[1]
                if op not in limits:
                    continue
                end = cases[i+1].start() if i+1 < len(cases) else len(executor)
                body = executor[case.end():end]
                with self.subTest(backend=name, command=op):
                    self.assertNotRegex(body, r'r->f\b')
                    for arg in re.findall(r'r->x\[(\d+)\]', body):
                        self.assertLessEqual(8 + (int(arg)+1)*8, limits[op])
                    self.assertNotRegex(body, r'r->x\[(?!\d+\])')
                checked.add(op)
        self.assertTrue(set(limits) <= checked, set(limits) - checked)


if __name__ == '__main__':
    unittest.main()
