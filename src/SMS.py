from twilio.rest import Client

from twilio.rest import Client



class SMS:
    account_sid = ''
    auth_token = ''
    user_phoneNum = "+8201049998216"
    twilio_phoneNum = ""

    @classmethod
    def sendMSG(cls, msg):
        try:
            client = Client(cls.account_sid, cls.auth_token)
            message = client.messages.create(to=cls.user_phoneNum, from_=cls.twilio_phoneNum, body=msg)
        except Exception as e:
            print("Twilio Error! : "+ e)




