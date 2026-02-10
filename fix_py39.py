#!/usr/bin/env python3
"""Fix Python 3.9 compatibility issues in the backend code."""

import os
import re

def fix_python39_compat():
    """Fix Union type annotations for Python 3.9."""
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, 'r') as file:
                    content = file.read()
                
                original = content
                
                # Fix Union types - add Union import if needed
                if 'str | Path' in content:
                    if 'from typing import' not in content:
                        content = re.sub(
                            r'from typing import (.*)',
                            r'from typing import \1, Union',
                            content
                        )
                    content = content.replace('str | Path', 'Union[str, Path]')
                    print(f'Fixed: {path}')
                
                # Ensure Union is imported when used
                if 'Union[' in content and 'from typing import' in content:
                    if 'Union' not in content.split('from typing import')[1].split('\n')[0]:
                        content = re.sub(
                            r'from typing import (.*)',
                            r'from typing import \1, Union',
                            content
                        )
                        print(f'Added Union import: {path}')
                
                if content != original:
                    with open(path, 'w') as file:
                        file.write(content)

if __name__ == '__main__':
    fix_python39_compat()
    print("Done fixing Python 3.9 compatibility!")

