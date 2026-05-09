# core/entry_gate.py

def entry_gate(portfolio):
    """
    Enforces max concurrent positions.
    """
    if not portfolio.can_enter():
        return False, "MAX_POSITION_REACHED"
    return True, "APPROVED"
