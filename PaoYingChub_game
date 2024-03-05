import pandas as pd
import random as rd
import time

# setting
result = {
    "win" : 0,
    "lose" : 0,
    "draw" : 0
}
rps = ["rock","paper","scissors"]


def RockPaperScissors():
    print("""Welcome to PaoYingChub!
Do you want to play with me?""")
    greeting = input("[Yes:No] ")

    if greeting.lower() == "yes":
          print("""Everyone called me Mr.Josh!
There are 2 rules in this game.
first is, there are 3 choices for you, which is rock, paper, and scissors!
and the second rule is, if you want to stop this game, say enough!""")
          time.sleep(0.3)
          print("Are you ready?")
          ans_start = input("[Yes:No] :")
          if(ans_start.lower() == "yes") :
            while True :
              print(""".
Pao Ying Chub!""")
              ans = input("Choose your!: ")
              bot_ans = rd.choice(rps)
              print(f"you : {ans}")
              if (ans.lower() == "rock") & (bot_ans == "rock"):
                print(f"mr.josh : {bot_ans} !")
                print("we draw!")
                result["draw"] = result["draw"]+1
              elif (ans.lower() == "rock") & (bot_ans == "paper"):
                print(f"mr.josh : {bot_ans} !")
                print("you win!")
                result["win"] = result["win"]+1
              elif (ans.lower() == "rock") & (bot_ans == "scissors"):
                print(f"mr.josh : {bot_ans} !")
                print("you lose!")
                result["lose"] = result["lose"]+1
              elif (ans.lower() == "paper") & (bot_ans == "rock"):
                print(f"mr.josh : {bot_ans} !")
                print("you win!")
                result["win"] = result["win"]+1
              elif (ans.lower() == "paper") & (bot_ans == "paper"):
                print(f"mr.josh : {bot_ans} !")
                print("we draw!")
                result["draw"] = result["draw"]+1
              elif (ans.lower() == "paper") & (bot_ans == "scissors"):
                print(f"mr.josh : {bot_ans} !")
                print("you lose!")
                result["lose"] = result["lose"]+1
              elif ((ans.lower() == "scissors")|(ans.lower() == "scissor")) & (bot_ans == "rock"):
                print(f"mr.josh : {bot_ans} !")
                print("you lose!")
                if(ans.lower() == "scissor"):
                  print("but, scissors have s, idiot!")
                result["lose"] = result["lose"]+1
              elif ((ans.lower() == "scissors")|(ans.lower() == "scissor")) & (bot_ans == "paper"):
                print(f"mr.josh : {bot_ans} !")
                print("you win!")
                if(ans.lower() == "scissor"):
                  print("but, scissors have s, idiot!")
                result["win"] = result["win"]+1
              elif ((ans.lower() == "scissors")|(ans.lower() == "scissor")) & (bot_ans == "scissors"):
                print(f"mr.josh : {bot_ans} !")
                print("we draw!")
                if(ans.lower() == "scissor"):
                  print("but, scissors have s, idiot!")
                result["draw"] = result["draw"]+1
              elif (ans.lower() == "enough!") | (ans.lower() == "enough"):
                print(""".
This is your honor!""")
                print(f"win : {result['win']}")
                print(f"lose : {result['lose']}")
                print(f"draw : {result['draw']}")
                if(result['win'] > (result['lose'] | result['draw'])) :
                 print("you are good!")
                if(result['win'] <= (result['lose'] | result['draw'])) :
                 print("you are bad!")
                break
              else :
                print("""that is not your choices!
check your spelling please!
next turn!""")

    else:
      print("See you next time!")
