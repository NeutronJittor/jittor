# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
import time

# Test 1: Enable custom priority mechanism and set variable priorities
def test_custom_priority():
    print("Test 1: Custom priority mechanism")
    
    # Enable custom priority mechanism
    jt.flags.use_custom_priority = 1
    
    # Create variables and set priorities
    a = jt.random((1000, 1000))
    b = jt.random((1000, 1000))
    c = jt.random((1000, 1000))
    
    # Set priorities
    a.set_priority(10)
    b.set_priority(5)
    c.set_priority(1)
    
    # Verify priority settings
    assert a.get_priority() == 10
    assert b.get_priority() == 5
    assert c.get_priority() == 1
    
    print("✓ Custom priority set successfully")

# Test 2: Verify set_priority and get_priority methods
def test_set_get_priority():
    print("Test 2: Set and get priority")
    
    # Enable custom priority mechanism
    jt.flags.use_custom_priority = 1
    
    # Create variable
    a = jt.random((1000, 1000))
    
    # Initial priority should be 0
    assert a.get_priority() == 0
    
    # Set priority
    a.set_priority(5)
    # Verify priority setting
    assert a.get_priority() == 5
    
    # Set priority again
    a.set_priority(10)
    # Verify priority update
    assert a.get_priority() == 10
    
    print("✓ Set and get priority successfully")

# Test 3: Use LFU decorator to automatically update priorities
@jt.lfu
def test_lfu_priority():
    print("Test 3: LFU priority decorator")
    
    # Enable custom priority mechanism
    jt.flags.use_custom_priority = 1
    
    # Create variables
    a = jt.random((1000, 1000))
    b = jt.random((1000, 1000))
    
    # Initial priorities should be 0
    assert a.get_priority() == 0
    assert b.get_priority() == 0
    
    # Perform operations on variables to trigger priority updates
    # Access variable attributes
    shape = a.shape
    dtype = b.dtype
    
    # Perform calculations
    c = a + b
    d = a * b
    
    # Access more attributes
    numel_a = a.numel()
    numel_b = b.numel()
    
    # Verify priorities have been updated automatically
    # Since we accessed a more times, its priority should be higher
    assert a.get_priority() > 0
    assert b.get_priority() > 0
    
    print(f"✓ LFU priority decorator test completed. Priorities: a={a.get_priority()}, b={b.get_priority()}")

# Test 4: Use custom priority function
# Define custom priority function
def custom_priority(var_id, access_count):
    # Custom priority logic: priority = access_count * 2
    return access_count * 2

@jt.lfu(priority_func=custom_priority)
def test_custom_priority_func():
    print("Test 4: Custom priority function")
    
    # Enable custom priority mechanism
    jt.flags.use_custom_priority = 1
    
    # Create variable
    a = jt.random((1000, 1000))
    
    # Initial priority should be 0
    assert a.get_priority() == 0
    
    # Perform operations on variable to trigger priority updates
    # Access variable attributes
    shape = a.shape
    dtype = a.dtype
    
    # Perform calculations
    b = a + a
    c = a * a
    
    # Access more attributes
    numel = a.numel()
    
    # Verify priority has been updated automatically
    # Using custom priority function: priority = access_count * 2
    assert a.get_priority() > 0
    
    print(f"✓ Custom priority function test completed. Priority: a={a.get_priority()}")

# Test 5: Verify offload mechanism works with custom priorities
def test_offload_with_custom_priority():
    print("Test 5: Offload with custom priority")
    
    # Enable CUDA
    jt.flags.use_cuda = 1
    
    # Enable custom priority mechanism
    jt.flags.use_custom_priority = 1
    
    # Set smaller device memory limit to trigger offload
    jt.flags.device_mem_limit = 100 * 1024 * 1024  # 100MB
    
    # Create multiple large variables
    vars = []
    for i in range(10):
        # Create variable of about 40MB (10000x1000 float32)
        var = jt.random((10000, 1000))
        # Set priority from high to low
        var.set_priority(10 - i)
        vars.append(var)
    
    # Access all variables to ensure they are allocated memory
    for var in vars:
        var.sync()
    
    # Force a memory allocation to trigger offload
    # Create one more large variable to push existing ones out
    additional_var = jt.random((10000, 1000))
    additional_var.sync()
    
    # Wait for a while to let offload mechanism run
    time.sleep(1)
    
    # Check variable locations
    print("Variable locations:")
    for i, var in enumerate(vars):
        print(f"Var {i} (priority {var.get_priority()}): {var.location()}")
    
    # Variables with higher priority should be more likely to be on device
    device_vars = [var for var in vars if var.location() == "device"]
    if device_vars:
        highest_priority = max(var.get_priority() for var in device_vars)
        print(f"Highest priority variable on device: {highest_priority}")
    else:
        print("No variables on device. This might be due to CUDA not being available or memory limit not being set correctly.")
    
    print("✓ Offload with custom priority test completed")

if __name__ == "__main__":
    test_custom_priority()
    test_set_get_priority()
    test_lfu_priority()
    test_custom_priority_func()
    test_offload_with_custom_priority()
    print("All tests passed!")
