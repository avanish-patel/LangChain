from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int


person: Person = Person(name="John", age=27)

print(person)

