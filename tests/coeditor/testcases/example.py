from dataclasses import dataclass


def foo(x, y):
    return x.bar().go(y.bar())


@dataclass
class Foo:
    value: float


    def bar(self):
        return Weight(self.value)


@dataclass
class Weight:
    value: float

    def go(self, other):
        return self.value + other.value


foo(Foo(1), Foo(2))
