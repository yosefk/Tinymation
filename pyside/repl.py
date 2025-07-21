import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QTextEdit, QLineEdit,
                               QScrollArea, QLabel, QDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
import traceback
from io import StringIO

class OutputRedirector:
    """Redirects stdout/stderr to the REPL output area"""
    def __init__(self, output_area):
        self.output_area = output_area
        self.text = ''

    def write(self, text):
        self.text += text
        try:
            newline = self.text.rindex('\n')
            to_append = self.text[:newline]
            self.text = self.text[newline+1:]
        except ValueError:
            return

        if to_append.strip():  # Only append non-empty text
            self.output_area.append(to_append.rstrip())
            # Scroll to bottom
            self.output_area.verticalScrollBar().setValue(
                self.output_area.verticalScrollBar().maximum()
            )

    def flush(self):
        pass  # Required for file-like object interface

class REPLDialog(QDialog):
    def __init__(self, parent=None, namespace=globals(), on_close=lambda: None):
        self.on_close = on_close
        super().__init__(parent)
        self.setWindowTitle("Python REPL")
        self.setGeometry(200, 200, 800, 600)

        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create the main layout
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create the output area (read-only text edit with scroll)
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setFont(QFont("Consolas", 10))
        self.output_area.append("Python REPL - Enter commands below\n" + "="*40)

        # Create output redirector
        self.output_redirector = OutputRedirector(self.output_area)

        # Create input area
        input_layout = QVBoxLayout()
        input_layout.setSpacing(5)

        # Command prompt label
        prompt_label = QLabel("Enter Python commands (Ctrl+Enter or Shift+Enter to execute):")
        prompt_label.setFont(QFont("Consolas", 10))

        # Command input area (multi-line)
        self.command_input = QTextEdit()
        self.command_input.setFont(QFont("Consolas", 10))
        self.command_input.setMinimumHeight(80)
        self.command_input.setMaximumHeight(150)
        self.command_input.setPlaceholderText("Enter Python code here...")

        # Button layout
        button_input_layout = QHBoxLayout()

        # Execute button
        execute_btn = QPushButton("Execute (Ctrl+Enter)")
        execute_btn.clicked.connect(self.execute_command)

        button_input_layout.addWidget(execute_btn)
        button_input_layout.addStretch()

        input_layout.addWidget(prompt_label)
        input_layout.addWidget(self.command_input)
        input_layout.addLayout(button_input_layout)

        # Button layout
        button_layout = QHBoxLayout()

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_output)

        # Quit button
        quit_btn = QPushButton("Close")
        quit_btn.clicked.connect(self.accept)  # Use accept() instead of close()

        button_layout.addWidget(clear_btn)
        button_layout.addStretch()
        button_layout.addWidget(quit_btn)

        # Add all components to main layout with proper stretch factors
        layout.addWidget(self.output_area, 1)  # Takes most space
        layout.addLayout(input_layout, 0)      # Fixed size
        layout.addLayout(button_layout, 0)     # Fixed size

        self.setLayout(layout)

        # Set focus to input field
        self.command_input.setFocus()

        # Add keyboard shortcuts
        self.command_input.keyPressEvent = self.handle_key_press

        self.namespace = namespace

        # Redirect stdout/stderr
        self.redirect_output()

        # History setup
        self.history = []
        self.search_prefix = None
        self.history_index = 0

    def redirect_output(self):
        """Redirect stdout and stderr to the REPL output"""
        sys.stdout = self.output_redirector
        sys.stderr = self.output_redirector

    def restore_output(self):
        """Restore original stdout and stderr"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def handle_key_press(self, event):
        # Handle Ctrl+Enter and Shift+Enter to execute command
        if (event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter):
            if event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier):
                self.execute_command()
                return

        # Handle history navigation
        if event.key() == Qt.Key_Up and event.modifiers() == Qt.ControlModifier:
            self.history_up()
            return
        elif event.key() == Qt.Key_Down and event.modifiers() == Qt.ControlModifier:
            self.history_down()
            return
        else:
            # Reset history navigation on other key presses
            if self.search_prefix is not None:
                self.search_prefix = None
            # Call the original keyPressEvent for normal behavior
            QTextEdit.keyPressEvent(self.command_input, event)

    def history_up(self):
        if self.search_prefix is None:
            self.search_prefix = self.command_input.toPlainText()
            self.history_index = len(self.history)
        # Find previous matching entry
        for i in range(self.history_index - 1, -1, -1):
            if self.history[i].startswith(self.search_prefix):
                self.history_index = i
                self.command_input.setPlainText(self.history[i])
                cursor = self.command_input.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.command_input.setTextCursor(cursor)
                return

    def history_down(self):
        if self.search_prefix is None:
            self.search_prefix = self.command_input.toPlainText()
            self.history_index = len(self.history)
        # Find next matching entry
        for i in range(self.history_index + 1, len(self.history)):
            if self.history[i].startswith(self.search_prefix):
                self.history_index = i
                self.command_input.setPlainText(self.history[i])
                cursor = self.command_input.textCursor()
                cursor.movePosition(QTextCursor.End)
                self.command_input.setTextCursor(cursor)
                return
        # If no more, return to the original prefix
        self.history_index = len(self.history)
        self.command_input.setPlainText(self.search_prefix)
        cursor = self.command_input.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.command_input.setTextCursor(cursor)

    def execute_command(self):
        command = self.command_input.toPlainText().strip()
        if not command:
            return

        full_command = self.command_input.toPlainText()
        # Add to history
        if full_command in self.history:
            self.history.remove(full_command)
        self.history.append(full_command)

        # Display the command with proper indentation
        command_lines = command.split('\n')
        for i, line in enumerate(command_lines):
            if i == 0:
                self.output_area.append(f">>> {line}")
            else:
                self.output_area.append(f"... {line}")

        try:
            # Try to compile as an expression first
            try:
                compiled = compile(command, '<repl>', 'eval')
                result = eval(compiled, self.namespace)
                if result is not None:
                    self.output_area.append(str(result))
            except SyntaxError:
                # If it's not an expression, execute as statement(s)
                compiled = compile(command, '<repl>', 'exec')
                exec(compiled, self.namespace)

        except Exception as e:
            # Display error message
            error_msg = f"Error: {type(e).__name__}: {str(e)}"
            self.output_area.append(error_msg)

            # Show traceback for debugging
            traceback_msg = traceback.format_exc()
            self.output_area.append(traceback_msg)

        # Clear input field
        self.command_input.clear()

        # Scroll to bottom
        self.output_area.verticalScrollBar().setValue(
            self.output_area.verticalScrollBar().maximum()
        )

    def clear_output(self):
        self.output_area.clear()
        self.output_area.append("Python REPL - Enter commands below\n" + "="*40)

    def accept(self):
        """Handle dialog acceptance (closing)"""
        self.restore_output()
        super().accept()
        self.on_close()

    def reject(self):
        """Handle dialog rejection (ESC key, X button)"""
        self.restore_output()
        super().reject()
        self.on_close()

    def closeEvent(self, event):
        """Handle window closing to restore stdout/stderr"""
        self.restore_output()
        event.accept()
        self.on_close()
