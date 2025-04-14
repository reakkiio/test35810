from json import loads
from re import findall
from time import sleep
from requests import Session
from webscout.litagent import LitAgent 

class Emailnator:
    def __init__(self) -> None:
        self.client = Session()
        self.client.get("https://www.emailnator.com/", timeout=6)
        self.cookies = self.client.cookies.get_dict()
        self.user_agent = LitAgent()
        self.client.headers = {
            "authority": "www.emailnator.com",
            "origin": "https://www.emailnator.com",
            "referer": "https://www.emailnator.com/",
            "user-agent": self.user_agent.random(),
            "x-xsrf-token": self.client.cookies.get("XSRF-TOKEN")[:-3] + "=",
        }

        self.email = None

    def get_mail(self):
        response = self.client.post(
            "https://www.emailnator.com/generate-email",
            json={
                "email": [
                    # "domain",
                    "plusGmail",
                    "dotGmail",
                ]
            },
        )

        self.email = loads(response.text)["email"][0]
        return self.email

    def get_message(self):
        print("waiting for code...")

        while True:
            sleep(2)
            mail_token = self.client.post(
                "https://www.emailnator.com/message-list", json={"email": self.email}
            )

            mail_token = loads(mail_token.text)["messageData"]

            if len(mail_token) == 2:
                print(mail_token[1]["messageID"])
                break

        mail_context = self.client.post(
            "https://www.emailnator.com/message-list",
            json={
                "email": self.email,
                "messageID": mail_token[1]["messageID"],
            },
        )

        return mail_context.text

    def get_verification_code(self):
        return findall(r';">(\d{6,7})</div>', self.get_message())[0]

if __name__ == "__main__":
    emailnator = Emailnator()
    r = emailnator.get_mail()
    print(r)
    print(emailnator.get_message())