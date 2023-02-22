################################################################################
# Set a breakpoint

def function_to_debug():
    x = 2

    # set breakpoint
    import pdb

    pdb.set_trace()
    y = x ** 2


function_to_debug()
