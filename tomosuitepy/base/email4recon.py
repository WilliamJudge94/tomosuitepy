from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import smtplib
import imaplib
import email


def send_email(recipient,
                            image,
                            subject,
                            body,
                            gmail_user,
                            gmail_pass):
  
  
    """
    Allow users to send an email.
    
    Parameters
    ----------
    recipient : list of str
        A list of all email addresses to send emails to.
    image : nd.array
        Image data to send in the email.
    subject : str
        The subject of the email.
    body : str
        The body of the email.
    gmail_user : str
        The email address of the sender.
    gmail_pass : str
        The email senders password.
 
    Returns
    -------
    None
        Sends email to designated addresses.
    """


    # Create message container.
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    msgRoot['From'] = gmail_user
    msgRoot['To'] = recipient

    # Create the body of the message.
    html = """\
        <p><br/>""" +  '<br/> <br/>' + body + """<br/>
            <img src="cid:image1">

        </p>
    """

    # Record the MIME types.
    msgHtml = MIMEText(html, 'html')
    
    msgImg = MIMEImage(image, 'png')
    msgImg.add_header('Content-ID', '<image1>')

    msgRoot.attach(msgHtml)
    msgRoot.attach(msgImg)

    # Send the message via local SMTP server.
    s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    s.ehlo()
    s.login(gmail_user, gmail_pass)
    s.sendmail(gmail_user, recipient, msgRoot.as_string())
    s.quit()
