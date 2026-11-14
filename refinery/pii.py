import ipaddress
import re

EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![\w-])"
)
IPV4 = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\.?\d)(?!\w)"
)
PHONE_NA = re.compile(r"(?<![\w+])(?:\+?1[ .-]?)?(?:\([2-9]\d{2}\)[ .-]?|[2-9]\d{2}[ .-])[2-9]\d{2}[ .-]\d{4}(?!\w)")
PHONE_INTL = re.compile(r"(?<![\w+])\+[1-9]\d{0,2}(?:[ .-]\(?\d{1,5}\)?){2,5}(?!\w)")

EMAIL_SUB = "email@example.com"
IP_SUB = "192.0.2.1"
PHONE_SUB = "555-0100"


def _public_ip(text):
    try:
        return ipaddress.ip_address(text).is_global
    except ValueError:
        return False


def _intl_phone(text):
    return 8 <= sum(ch.isdigit() for ch in text) <= 15


def redact(text, emails=True, ips=True, phones=True):
    counts = {"emails": 0, "ips": 0, "phones": 0}

    def swap(kind, sub, valid=None):
        def inner(match):
            if valid is not None and not valid(match.group(0)):
                return match.group(0)
            counts[kind] += 1
            return sub

        return inner

    if emails and "@" in text:
        text = EMAIL.sub(swap("emails", EMAIL_SUB), text)
    if ips and "." in text:
        text = IPV4.sub(swap("ips", IP_SUB, _public_ip), text)
    if phones:
        text = PHONE_NA.sub(swap("phones", PHONE_SUB), text)
        if "+" in text:
            text = PHONE_INTL.sub(swap("phones", PHONE_SUB, _intl_phone), text)
    return text, counts
