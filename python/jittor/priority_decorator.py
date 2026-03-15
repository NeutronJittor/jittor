# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
import functools

# Global access count
access_counts = {}

class PriorityDecorator:
    """
    Priority decorator for automatically modifying priorities during variable operations
    
    Example:
    ```python
    # Use LFU algorithm
    @PriorityDecorator()
    def my_function():
        # Variable operations within the function will automatically update priorities
        pass
    
    # Use custom priority function
    def custom_priority_func(var_id, access_count):
        # Custom priority logic
        return access_count * 2
    
    @PriorityDecorator(priority_func=custom_priority_func)
    def my_function2():
        # Variable operations within the function will automatically update priorities
        pass
    ```
    """
    
    def __init__(self, lfu=True, priority_func=None):
        """
        Initialize priority decorator
        
        Args:
            lfu: Whether to use LFU (Least Frequently Used) algorithm
            priority_func: Custom priority function that takes var_id and access_count as parameters
        """
        self.lfu = lfu
        self.priority_func = priority_func
    
    def __call__(self, func):
        """
        Call decorator
        
        Args:
            func: Decorated function
        
        Returns:
            Decorated function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save original __getattribute__ method
            original_getattribute = jt.Var.__getattribute__
            lfu = self.lfu
            priority_func = self.priority_func
            
            def new_getattribute(self, name):
                # Update priority when accessing variable attributes or methods
                if (lfu or priority_func) and name not in ['__getattribute__', 'set_priority', 'get_priority']:
                    var_id = id(self)
                    access_counts[var_id] = access_counts.get(var_id, 0) + 1
                    if hasattr(self, 'set_priority'):
                        if priority_func:
                            # Use custom priority function
                            priority = priority_func(var_id, access_counts[var_id])
                        else:
                            # Use LFU algorithm
                            priority = access_counts[var_id]
                        self.set_priority(priority)
                # Call original __getattribute__ method
                return original_getattribute(self, name)
            
            # Replace __getattribute__ method
            jt.Var.__getattribute__ = new_getattribute
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Restore original __getattribute__ method
            jt.Var.__getattribute__ = original_getattribute
            
            return result
        
        return wrapper

# Global LFU decorator instance
lfu_decorator = PriorityDecorator(lfu=True)

# Convenience function
def lfu(func=None, priority_func=None):
    """
    LFU (Least Frequently Used) decorator for automatically updating priorities during variable operations
    
    Args:
        func: Decorated function
        priority_func: Custom priority function that takes var_id and access_count as parameters
    
    Returns:
        Decorated function
    """
    if func is None:
        # Return a decorator with custom priority function
        def decorator(func):
            return PriorityDecorator(lfu=True, priority_func=priority_func)(func)
        return decorator
    else:
        # Use default LFU
        return lfu_decorator(func)
