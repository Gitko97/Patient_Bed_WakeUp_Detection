from twilio.rest import Client

from twilio.rest import Client



class SMS:
    account_sid = 'ACde5158a4a27e8b99d712c554172c9bd3'
    auth_token = '01457c77692dc17509e78c375c3f851b'
    user_phoneNum = "+8201049998216"
    twilio_phoneNum = "+18326376336"

    @classmethod
    def sendMSG(cls, msg):
        try:
            client = Client(cls.account_sid, cls.auth_token)
            message = client.messages.create(to=cls.user_phoneNum, from_=cls.twilio_phoneNum, body=msg)
        except Exception as e:
            print("Twilio Error! : "+ e)




