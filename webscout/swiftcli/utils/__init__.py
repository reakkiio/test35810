"""Utility functions for SwiftCLI."""

from .formatting import (
    style_text,
    format_error,
    format_warning,
    format_success,
    format_info,
    create_table,
    truncate_text,
    wrap_text,
    format_dict,
    format_list,
    strip_ansi,
    get_terminal_size,
    clear_screen,
    create_padding
)

from .parsing import (
    parse_args,
    validate_required,
    convert_type,
    validate_choice,
    load_config_file,
    parse_key_value,
    parse_list,
    parse_dict,
    get_env_var
)

__all__ = [
    # Formatting utilities
    'style_text',
    'format_error',
    'format_warning',
    'format_success',
    'format_info',
    'create_table',
    'truncate_text',
    'wrap_text',
    'format_dict',
    'format_list',
    'strip_ansi',
    'get_terminal_size',
    'clear_screen',
    'create_padding',
    
    # Parsing utilities
    'parse_args',
    'validate_required',
    'convert_type',
    'validate_choice',
    'load_config_file',
    'parse_key_value',
    'parse_list',
    'parse_dict',
    'get_env_var'
]
