#! /usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-unsafe

import code
import sys

if len(sys.argv) > 1:
    f = open(sys.argv[1])
    x = f.read()
    f.close()
    exec(x)

code.interact(local=locals())
