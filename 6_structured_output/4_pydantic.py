from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Default Name"
    age: Optional[int] = None
    # validate email if exist to be correct email format
    email: Optional[EmailStr] = None
    # validate gpa to be in between 0 and 4 if exist
    gpa: Optional[float] = Field(default=0, gt=0, lt=4, description="GPA")

new_student = Student(name="Avinash", age=28, email="test@email.com", gpa=3.14)

print(new_student)
print(type(new_student))

student_dict = dict(new_student)
print(student_dict)
print(student_dict["name"])

name_dict = {"name":"Dixit"}
another_student = Student(**name_dict)
print(another_student)

default_student = Student()
print(default_student)