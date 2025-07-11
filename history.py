# advanced-world-editor/history.py

class HistoryManager:
    """Manages the undo/redo stack for application actions."""
    def __init__(self, app):
        self.app = app
        self.undo_stack = []
        self.redo_stack = []

    def push(self, action):
        self.undo_stack.append(action)
        self.redo_stack.clear()
        self.app.update_edit_menu_state()

    def undo(self):
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        action.undo()
        self.redo_stack.append(action)
        self.app.update_edit_menu_state()
        self.app.on_history_change()

    def redo(self):
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        action.redo()
        self.undo_stack.append(action)
        self.app.update_edit_menu_state()
        self.app.on_history_change()


class PaintAction:
    """An undoable action representing a paint stroke."""
    def __init__(self, data_map, indices, before, after):
        self.data_map, self.indices, self.before, self.after = (
            data_map,
            indices,
            before.copy(),
            after.copy(),
        )

    def undo(self):
        self.data_map[self.indices] = self.before

    def redo(self):
        self.data_map[self.indices] = self.after