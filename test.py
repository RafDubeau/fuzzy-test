
class ExampleClass:

    def __init__(self, name):
        self.name = name
    
    def count_letters(self):
        return len(self.name)

if __name__ == "__main__":
    
    example1 = ExampleClass("Christian")
    example2 = ExampleClass("Rafael")

    print(example1.count_letters())
    print(example2.count_letters())