"""
ZeroArt: A zero-dependency ASCII art text generator

Create awesome ASCII art text without external dependencies!
"""

from typing import Dict, List, Literal, Optional, Union
from .base import ZeroArtFont
from .fonts import BlockFont, SlantFont, NeonFont, CyberFont, DottedFont, ShadowFont
from .effects import AsciiArtEffects

FontType = Literal['block', 'slant', 'neon', 'cyber', 'dotted', 'shadow']

def figlet_format(text: str, font: Union[str, ZeroArtFont] = 'block') -> str:
    """
    Generate ASCII art text
    
    :param text: Text to convert
    :param font: Font style (default: 'block')
    :return: ASCII art representation of text
    """
    font_map: Dict[str, ZeroArtFont] = {
        'block': BlockFont(),
        'slant': SlantFont(),
        'neon': NeonFont(),
        'cyber': CyberFont(),
        'dotted': DottedFont(),
        'shadow': ShadowFont()
    }
    
    if isinstance(font, str):
        selected_font: ZeroArtFont = font_map.get(font.lower(), BlockFont())
    else:
        selected_font = font
    return selected_font.render(text)

def print_figlet(text: str, font: Union[str, ZeroArtFont] = 'block') -> None:
    """
    Print ASCII art text directly
    
    :param text: Text to convert and print
    :param font: Font style (default: 'block')
    """
    print(figlet_format(text, font))

# Expose additional effects with font handling
def rainbow(text: str, font: Union[str, ZeroArtFont] = 'block') -> str:
    """
    Apply a rainbow-like color effect to ASCII art
    
    :param text: Text to render
    :param font: Font style (default: 'block')
    :return: Rainbow-styled ASCII art
    """
    font_map: Dict[str, ZeroArtFont] = {
        'block': BlockFont(),
        'slant': SlantFont(),
        'neon': NeonFont(),
        'cyber': CyberFont(),
        'dotted': DottedFont(),
        'shadow': ShadowFont()
    }
    
    if isinstance(font, str):
        selected_font: ZeroArtFont = font_map.get(font.lower(), BlockFont())
    else:
        selected_font = font
    return AsciiArtEffects.rainbow_effect(text, selected_font)

def glitch(text: str, font: Union[str, ZeroArtFont] = 'block', glitch_intensity: float = 0.1) -> str:
    """
    Apply a glitch-like distortion to ASCII art
    
    :param text: Text to render
    :param font: Font style (default: 'block')
    :param glitch_intensity: Probability of character distortion
    :return: Glitched ASCII art
    """
    font_map: Dict[str, ZeroArtFont] = {
        'block': BlockFont(),
        'slant': SlantFont(),
        'neon': NeonFont(),
        'cyber': CyberFont(),
        'dotted': DottedFont(),
        'shadow': ShadowFont()
    }
    
    if isinstance(font, str):
        selected_font: ZeroArtFont = font_map.get(font.lower(), BlockFont())
    else:
        selected_font = font
    return AsciiArtEffects.glitch_effect(text, selected_font, glitch_intensity)

wrap_text = AsciiArtEffects.wrap_text

def outline(text: str, font: Union[str, ZeroArtFont] = 'block', outline_char: str = '*') -> str:
    """
    Add an outline effect to ASCII art
    
    :param text: Text to render
    :param font: Font style (default: 'block')
    :param outline_char: Character to use for outline
    :return: ASCII art with outline
    """
    font_map: Dict[str, ZeroArtFont] = {
        'block': BlockFont(),
        'slant': SlantFont(),
        'neon': NeonFont(),
        'cyber': CyberFont(),
        'dotted': DottedFont(),
        'shadow': ShadowFont()
    }
    
    if isinstance(font, str):
        selected_font: ZeroArtFont = font_map.get(font.lower(), BlockFont())
    else:
        selected_font = font
    return AsciiArtEffects.outline_effect(text, selected_font, outline_char)

__all__ = [
    'figlet_format', 
    'print_figlet', 
    'rainbow', 
    'glitch', 
    'wrap_text', 
    'outline',
    'BlockFont', 
    'SlantFont', 
    'NeonFont', 
    'CyberFont',
    'DottedFont',
    'ShadowFont',
    'ZeroArtFont',
    'FontType'
]