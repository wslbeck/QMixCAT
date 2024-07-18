from typing import Optional, Any, List, Optional, Tuple, Type, TypeVar, Union
from configparser import ConfigParser

from shlex import shlex
import string

T = TypeVar("T")
class Config:
    def __init__(self):
        self.parser: ConfigParser = None  # type: ignore
    def load(self, path: Optional[str]):
        self.parser = ConfigParser()
        self.parser.read(path, encoding='utf-8')
        # print(self.parser.sections())
        # with open(path) as f:
        #     self.parser = ConfigParser()
        #     self.parser.read(f, encoding="utf-8")
        #     print(self.parser.sections())
    def __call__(
        self,
        option: str,
        default: Any = None,
        cast: Type[T] = str,
        save: bool = True,
        section: Optional[str] = None,
    ) -> T:
        if self.parser is None:
            raise ValueError("No configuration loaded")
        if self.parser.has_option(section, option):
            value = self.parser.get(section, option)
        elif self.parser.has_option(section.lower(), option):
            value = self.parser.get(section.lower(), option)
        elif default is None:
            raise ValueError("Value {} not found.".format(option))
        elif not self.allow_defaults and save:
            raise ValueError(f"Value '{option}' not found in config (defaults not allowed).")
        else:
            value = default
            if save:
                self.set(section, option, value)   
        return self.cast(value, cast)

    def cast(self, value, cast):
        # Do the casting to get the correct type
        if cast is bool:
            value = str(value).lower()
            if value in {"true", "yes", "y", "on", "1"}:
                return True  # type: ignore
            elif value in {"false", "no", "n", "off", "0"}:
                return False  # type: ignore
            raise ValueError("Parse error")
        return cast(value)


config = Config()

class Csv(object):
    """
    Produces a csv parser that return a list of transformed elements. From python-decouple.
    """

    def __init__(
        self, cast: Type[T] = str, delimiter=",", strip=string.whitespace, post_process=list
    ):
        """
        Parameters:
        cast -- callable that transforms the item just before it's added to the list.
        delimiter -- string of delimiters chars passed to shlex.
        strip -- string of non-relevant characters to be passed to str.strip after the split.
        post_process -- callable to post process all casted values. Default is `list`.
        """
        self.cast: Type[T] = cast
        self.delimiter = delimiter
        self.strip = strip
        self.post_process = post_process

    def __call__(self, value: Union[str, Tuple[T], List[T]]) -> List[T]:
        """The actual transformation"""
        if isinstance(value, (tuple, list)):
            # if default value is a list
            value = "".join(str(v) + self.delimiter for v in value)[:-1]

        def transform(s):
            return self.cast(s.strip(self.strip))

        splitter = shlex(value, posix=True)
        splitter.whitespace = self.delimiter
        splitter.whitespace_split = True

        return self.post_process(transform(s) for s in splitter)
