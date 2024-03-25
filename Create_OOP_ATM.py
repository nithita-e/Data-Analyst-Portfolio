class ATM:
    def __init__(self, atm_id, pin, name, cash_balance):
        self.atm_id = atm_id
        self.pin = pin
        self.name = name,
        self.cash_balance = cash_balance

    def deposit(self, atm_id):
        print("please, enter your pin.")
        user_pin = input("pin: ")
        if float(user_pin) == self.pin :
            print("Hello! How much do you want to deposit?")
            deposit = float(input("deposit: "))
            self.cash_balance = deposit + self.cash_balance
            print(f"Total balance = {self.cash_balance}")
            print("Have a nice day!")
        else :
          print("You've entered incorrect pin, please try again later.")

    def withdraw(self,atm_id):
        print("please, enter your pin.")
        user_pin = input("pin: ")
        if float(user_pin) == self.pin :
            print("Hello! How much do you want to withdraw?")
            withdraw = float(input("withdraw: "))
            self.cash_balance = self.cash_balance - withdraw
            print(f"Total balance = {self.cash_balance}")
            print("Have a nice day!")
        else :
          print("You've entered incorrect pin, please try again later.")
    
    def checkBalance(self,atm_id):
        print("please, enter your pin")
        user_pin = input("pin: ")
        if float(user_pin) == self.pin :
            print(f"Your total balance = {self.cash_balance}")
            print("Have a nice day!")
        else :
          print("You've entered incorrect pin, please try again later.")
