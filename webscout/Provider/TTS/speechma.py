##################################################################################
##  Modified version of code written by t.me/infip1217                          ##
##################################################################################
import time
import requests
import pathlib
import tempfile
from io import BytesIO
from webscout import exceptions
from webscout.litagent import LitAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
from webscout.Provider.TTS import utils
from webscout.Provider.TTS.base import BaseTTSProvider

class SpeechMaTTS(BaseTTSProvider):
    """
    Text-to-speech provider using the SpeechMa API.
    """
    # Request headers
    headers = {
        "accept": "*/*",
        "accept-language": "en-IN,en-GB;q=0.9,en-US;q=0.8,en;q=0.7,en-AU;q=0.6",
        "content-type": "application/json",
        "origin": "https://speechma.com",
        "priority": "u=1, i",
        "User-Agent": LitAgent().random()
    }

    # Available voices with their IDs
    all_voices = {
        # Multilingual voices
        "Andrew Multilingual": "voice-107",  # Male, Multilingual, United States
        "Ava Multilingual": "voice-110",  # Female, Multilingual, United States
        "Brian Multilingual": "voice-112",   # Male, Multilingual, United States
        "Emma Multilingual": "voice-115",    # Female, Multilingual, United States
        "Remy Multilingual": "voice-142",    # Male, Multilingual, France
        "Vivienne Multilingual": "voice-143", # Female, Multilingual, France
        "Florian Multilingual": "voice-154",  # Male, Multilingual, Germany
        "Seraphina Multilingual": "voice-157", # Female, Multilingual, Germany
        "Giuseppe Multilingual": "voice-177", # Male, Multilingual, Italy
        "Hyunsu Multilingual": "voice-189",   # Male, Multilingual, South Korea
        "Thalita Multilingual": "voice-222",  # Female, Multilingual, Brazil
        # English (US)
        "Ana": "voice-106",  # Female, English, United States
        "Andrew": "voice-108",  # Male, English, United States
        "Aria": "voice-109",  # Female, English, United States
        "Ava": "voice-111",  # Female, English, United States
        "Brian": "voice-113",  # Male, English, United States
        "Christopher": "voice-114",  # Male, English, United States
        "Emma": "voice-116",  # Female, English, United States
        "Eric": "voice-117",  # Male, English, United States
        "Guy": "voice-118",  # Male, English, United States
        "Jenny": "voice-119",  # Female, English, United States
        "Michelle": "voice-120",  # Female, English, United States
        "Roger": "voice-121",  # Male, English, United States
        "Steffan": "voice-122",  # Male, English, United States
        # English (UK)
        "Libby": "voice-82",  # Female, English, United Kingdom
        "Maisie": "voice-83",  # Female, English, United Kingdom
        "Ryan": "voice-84",  # Male, English, United Kingdom
        "Sonia": "voice-85",  # Female, English, United Kingdom
        "Thomas": "voice-86",  # Male, English, United Kingdom
        # English (Australia)
        "Natasha": "voice-78",  # Female, English, Australia
        "William": "voice-79",  # Male, English, Australia
        # English (Canada)
        "Clara": "voice-80",  # Female, English, Canada
        "Liam": "voice-81",  # Male, English, Canada
        # English (India)
        "Neerja Expressive": "voice-91",  # Female, English, India
        "Neerja": "voice-92",  # Female, English, India
        "Prabhat": "voice-93",  # Male, English, India
        # English (Hong Kong)
        "Sam": "voice-87",  # Male, English, Hong Kong
        "Yan": "voice-88",  # Female, English, Hong Kong
        # English (Ireland)
        "Connor": "voice-89",  # Male, English, Ireland
        "Emily": "voice-90",  # Female, English, Ireland
        # English (Kenya)
        "Asilia": "voice-94",  # Female, English, Kenya
        "Chilemba": "voice-95",  # Male, English, Kenya
        # English (Nigeria)
        "Abeo": "voice-96",  # Male, English, Nigeria
        "Ezinne": "voice-97",  # Female, English, Nigeria
        # English (New Zealand)
        "Mitchell": "voice-98",  # Male, English, New Zealand
        "Molly": "voice-99",  # Female, English, New Zealand
        # English (Philippines)
        "James": "voice-100",  # Male, English, Philippines
        "Rosa": "voice-101",  # Female, English, Philippines
        # English (Singapore)
        "Luna": "voice-102",  # Female, English, Singapore
        "Wayne": "voice-103",  # Male, English, Singapore
        # English (Tanzania)
        "Elimu": "voice-104",  # Male, English, Tanzania
        "Imani": "voice-105",  # Female, English, Tanzania
        # English (South Africa)
        "Leah": "voice-123",  # Female, English, South Africa
        "Luke": "voice-124",  # Male, English, South Africa
        # Spanish (Argentina)
        "Elena": "voice-239",  # Female, Spanish, Argentina
        "Tomas": "voice-240",  # Male, Spanish, Argentina
        # Spanish (Bolivia)
        "Marcelo": "voice-241",  # Male, Spanish, Bolivia
        "Sofia": "voice-242",  # Female, Spanish, Bolivia
        # Spanish (Chile)
        "Catalina": "voice-243",  # Female, Spanish, Chile
        "Lorenzo": "voice-244",  # Male, Spanish, Chile
        # Spanish (Colombia)
        "Gonzalo": "voice-245",  # Male, Spanish, Colombia
        "Salome": "voice-246",  # Female, Spanish, Colombia
        # Spanish (Costa Rica)
        "Juan": "voice-247",  # Male, Spanish, Costa Rica
        "Maria": "voice-248",  # Female, Spanish, Costa Rica
        # Spanish (Cuba)
        "Belkys": "voice-249",  # Female, Spanish, Cuba
        "Manuel": "voice-250",  # Male, Spanish, Cuba
        # Spanish (Dominican Republic)
        "Emilio": "voice-251",  # Male, Spanish, Dominican Republic
        "Ramona": "voice-252",  # Female, Spanish, Dominican Republic
        # Spanish (Ecuador)
        "Andrea": "voice-253",  # Female, Spanish, Ecuador
        "Luis": "voice-254",  # Male, Spanish, Ecuador
        # Spanish (Spain)
        "Alvaro": "voice-255",  # Male, Spanish, Spain
        "Elvira": "voice-256",  # Female, Spanish, Spain
        "Ximena": "voice-257",  # Female, Spanish, Spain
        # Spanish (Equatorial Guinea)
        "Javier": "voice-258",  # Male, Spanish, Equatorial Guinea
        "Teresa": "voice-259",  # Female, Spanish, Equatorial Guinea
        # Spanish (Guatemala)
        "Andres": "voice-260",  # Male, Spanish, Guatemala
        "Marta": "voice-261",  # Female, Spanish, Guatemala
        # Spanish (Honduras)
        "Carlos": "voice-262",  # Male, Spanish, Honduras
        "Karla": "voice-263",  # Female, Spanish, Honduras
        # Spanish (Mexico)
        "Dalia": "voice-264",  # Female, Spanish, Mexico
        "Jorge": "voice-265",  # Male, Spanish, Mexico
        # Spanish (Nicaragua)
        "Federico": "voice-266",  # Male, Spanish, Nicaragua
        "Yolanda": "voice-267",  # Female, Spanish, Nicaragua
        # Spanish (Panama)
        "Margarita": "voice-268",  # Female, Spanish, Panama
        "Roberto": "voice-269",  # Male, Spanish, Panama
        # Spanish (Peru)
        "Alex": "voice-270",  # Male, Spanish, Peru
        "Camila": "voice-271",  # Female, Spanish, Peru
        # Spanish (Puerto Rico)
        "Karina": "voice-272",  # Female, Spanish, Puerto Rico
        "Victor": "voice-273",  # Male, Spanish, Puerto Rico
        # Spanish (Paraguay)
        "Mario": "voice-274",  # Male, Spanish, Paraguay
        "Tania": "voice-275",  # Female, Spanish, Paraguay
        # Spanish (El Salvador)
        "Lorena": "voice-276",  # Female, Spanish, El Salvador
        "Rodrigo": "voice-277",  # Male, Spanish, El Salvador
        # Spanish (United States)
        "Alonso": "voice-278",  # Male, Spanish, United States
        "Paloma": "voice-279",  # Female, Spanish, United States
        # Spanish (Uruguay)
        "Mateo": "voice-280",  # Male, Spanish, Uruguay
        "Valentina": "voice-281",  # Female, Spanish, Uruguay
        # Spanish (Venezuela)
        "Paola": "voice-282",  # Female, Spanish, Venezuela
        "Sebastian": "voice-283",  # Male, Spanish, Venezuela
        # Chinese (China)
        "Xiaoxiao": "voice-53",  # Female, Chinese, China
        "Xiaoyi": "voice-54",  # Female, Chinese, China
        "Yunjian": "voice-55",  # Male, Chinese, China
        "Yunxi": "voice-56",  # Male, Chinese, China
        "Yunxia": "voice-57",  # Male, Chinese, China
        "Yunyang": "voice-58",  # Male, Chinese, China
        "Xiaobei": "voice-59",  # Female, Chinese, China
        "Xiaoni": "voice-60",  # Female, Chinese, China
        # Chinese (Hong Kong)
        "HiuGaai": "voice-61",  # Female, Chinese, Hong Kong
        "HiuMaan": "voice-62",  # Female, Chinese, Hong Kong
        "WanLung": "voice-63",  # Male, Chinese, Hong Kong
        # Chinese (Taiwan)
        "HsiaoChen": "voice-64",  # Female, Chinese, Taiwan
        "HsiaoYu": "voice-65",  # Female, Chinese, Taiwan
        "YunJhe": "voice-66",  # Male, Chinese, Taiwan
        # French (Belgium)
        "Charline": "voice-131",  # Female, French, Belgium
        "Gerard": "voice-132",  # Male, French, Belgium
        # French (Canada)
        "Antoine": "voice-133",  # Male, French, Canada
        "Jean": "voice-134",  # Male, French, Canada
        "Sylvie": "voice-135",  # Female, French, Canada
        "Thierry": "voice-136",  # Male, French, Canada
        # French (Switzerland)
        "Ariane": "voice-137",  # Female, French, Switzerland
        "Fabrice": "voice-138",  # Male, French, Switzerland
        # French (France)
        "Denise": "voice-139",  # Female, French, France
        "Eloise": "voice-140",  # Female, French, France
        "Henri": "voice-141",  # Male, French, France
        # German (Austria)
        "Ingrid": "voice-148",  # Female, German, Austria
        "Jonas": "voice-149",  # Male, German, Austria
        # German (Switzerland)
        "Jan": "voice-150",  # Male, German, Switzerland
        "Leni": "voice-151",  # Female, German, Switzerland
        # German (Germany)
        "Amala": "voice-152",  # Female, German, Germany
        "Conrad": "voice-153",  # Male, German, Germany
        "Katja": "voice-155",  # Female, German, Germany
        "Killian": "voice-156",  # Male, German, Germany
        # Arabic (United Arab Emirates)
        "Fatima": "voice-7",  # Female, Arabic, United Arab Emirates
        "Hamdan": "voice-8",  # Male, Arabic, United Arab Emirates
        # Arabic (Bahrain)
        "Ali": "voice-9",  # Male, Arabic, Bahrain
        "Laila": "voice-10",  # Female, Arabic, Bahrain
        # Arabic (Algeria)
        "Amina": "voice-11",  # Female, Arabic, Algeria
        "Ismael": "voice-12",  # Male, Arabic, Algeria
        # Arabic (Egypt)
        "Salma": "voice-13",  # Female, Arabic, Egypt
        "Shakir": "voice-14",  # Male, Arabic, Egypt
        # Arabic (Iraq)
        "Bassel": "voice-15",  # Male, Arabic, Iraq
        "Rana": "voice-16",  # Female, Arabic, Iraq
        # Arabic (Jordan)
        "Sana": "voice-17",  # Female, Arabic, Jordan
        "Taim": "voice-18",  # Male, Arabic, Jordan
        # Arabic (Kuwait)
        "Fahed": "voice-19",  # Male, Arabic, Kuwait
        "Noura": "voice-20",  # Female, Arabic, Kuwait
        # Arabic (Lebanon)
        "Layla": "voice-21",  # Female, Arabic, Lebanon
        "Rami": "voice-22",  # Male, Arabic, Lebanon
        # Arabic (Libya)
        "Iman": "voice-23",  # Female, Arabic, Libya
        "Omar": "voice-24",  # Male, Arabic, Libya
        # Arabic (Morocco)
        "Jamal": "voice-25",  # Male, Arabic, Morocco
        "Mouna": "voice-26",  # Female, Arabic, Morocco
        # Arabic (Oman)
        "Abdullah": "voice-27",  # Male, Arabic, Oman
        "Aysha": "voice-28",  # Female, Arabic, Oman
        # Arabic (Qatar)
        "Amal": "voice-29",  # Female, Arabic, Qatar
        "Moaz": "voice-30",  # Male, Arabic, Qatar
        # Arabic (Saudi Arabia)
        "Hamed": "voice-31",  # Male, Arabic, Saudi Arabia
        "Zariyah": "voice-32",  # Female, Arabic, Saudi Arabia
        # Arabic (Syria)
        "Amany": "voice-33",  # Female, Arabic, Syria
        "Laith": "voice-34",  # Male, Arabic, Syria
        # Arabic (Tunisia)
        "Hedi": "voice-35",  # Male, Arabic, Tunisia
        "Reem": "voice-36",  # Female, Arabic, Tunisia
        # Arabic (Yemen)
        "Maryam": "voice-37",  # Female, Arabic, Yemen
        "Saleh": "voice-38",  # Male, Arabic, Yemen
        # Afrikaans (South Africa)
        "Adri": "voice-1",  # Female, Afrikaans, South Africa
        "Willem": "voice-2",  # Male, Afrikaans, South Africa
        # Albanian (Albania)
        "Anila": "voice-3",  # Female, Albanian, Albania
        "Ilir": "voice-4",  # Male, Albanian, Albania
        # Amharic (Ethiopia)
        "Ameha": "voice-5",  # Male, Amharic, Ethiopia
        "Mekdes": "voice-6",  # Female, Amharic, Ethiopia
        # Azerbaijani (Azerbaijan)
        "Babek": "voice-39",  # Male, Azerbaijani, Azerbaijan
        "Banu": "voice-40",  # Female, Azerbaijani, Azerbaijan
        # Bengali (Bangladesh)
        "Nabanita": "voice-41",  # Female, Bengali, Bangladesh
        "Pradeep": "voice-42",  # Male, Bengali, Bangladesh
        # Bengali (India)
        "Bashkar": "voice-43",  # Male, Bengali, India
        "Tanishaa": "voice-44",  # Female, Bengali, India
        # Bosnian (Bosnia and Herzegovina)
        "Goran": "voice-45",  # Male, Bosnian, Bosnia and Herzegovina
        "Vesna": "voice-46",  # Female, Bosnian, Bosnia and Herzegovina
        # Bulgarian (Bulgaria)
        "Borislav": "voice-47",  # Male, Bulgarian, Bulgaria
        "Kalina": "voice-48",  # Female, Bulgarian, Bulgaria
        # Burmese (Myanmar)
        "Nilar": "voice-49",  # Female, Burmese, Myanmar
        "Thiha": "voice-50",  # Male, Burmese, Myanmar
        # Catalan (Spain)
        "Enric": "voice-51",  # Male, Catalan, Spain
        "Joana": "voice-52",  # Female, Catalan, Spain
        # Croatian (Croatia)
        "Gabrijela": "voice-67",  # Female, Croatian, Croatia
        "Srecko": "voice-68",  # Male, Croatian, Croatia
        # Czech (Czech Republic)
        "Antonin": "voice-69",  # Male, Czech, Czech Republic
        "Vlasta": "voice-70",  # Female, Czech, Czech Republic
        # Danish (Denmark)
        "Christel": "voice-71",  # Female, Danish, Denmark
        "Jeppe": "voice-72",  # Male, Danish, Denmark
        # Dutch (Belgium)
        "Arnaud": "voice-73",  # Male, Dutch, Belgium
        "Dena": "voice-74",  # Female, Dutch, Belgium
        # Dutch (Netherlands)
        "Colette": "voice-75",  # Female, Dutch, Netherlands
        "Fenna": "voice-76",  # Female, Dutch, Netherlands
        "Maarten": "voice-77",  # Male, Dutch, Netherlands
        # Estonian (Estonia)
        "Anu": "voice-125",  # Female, Estonian, Estonia
        "Kert": "voice-126",  # Male, Estonian, Estonia
        # Filipino (Philippines)
        "Angelo": "voice-127",  # Male, Filipino, Philippines
        "Blessica": "voice-128",  # Female, Filipino, Philippines
        # Finnish (Finland)
        "Harri": "voice-129",  # Male, Finnish, Finland
        "Noora": "voice-130",  # Female, Finnish, Finland
        # Galician (Spain)
        "Roi": "voice-144",  # Male, Galician, Spain
        "Sabela": "voice-145",  # Female, Galician, Spain
        # Georgian (Georgia)
        "Eka": "voice-146",  # Female, Georgian, Georgia
        "Giorgi": "voice-147",  # Male, Georgian, Georgia
        # Greek (Greece)
        "Athina": "voice-158",  # Female, Greek, Greece
        "Nestoras": "voice-159",  # Male, Greek, Greece (Note: voice-160 is a duplicate name)
        # Gujarati (India)
        "Dhwani": "voice-161",  # Female, Gujarati, India
        "Niranjan": "voice-162",  # Male, Gujarati, India
        # Hebrew (Israel)
        "Avri": "voice-163",  # Male, Hebrew, Israel
        "Hila": "voice-164",  # Female, Hebrew, Israel
        # Hindi (India)
        "Madhur": "voice-165",  # Male, Hindi, India
        "Swara": "voice-166",  # Female, Hindi, India
        # Hungarian (Hungary)
        "Noemi": "voice-167",  # Female, Hungarian, Hungary
        "Tamas": "voice-168",  # Male, Hungarian, Hungary
        # Icelandic (Iceland)
        "Gudrun": "voice-169",  # Female, Icelandic, Iceland
        "Gunnar": "voice-170",  # Male, Icelandic, Iceland
        # Indonesian (Indonesia)
        "Ardi": "voice-171",  # Male, Indonesian, Indonesia
        "Gadis": "voice-172",  # Female, Indonesian, Indonesia
        # Irish (Ireland)
        "Colm": "voice-173",  # Male, Irish, Ireland
        "Orla": "voice-174",  # Female, Irish, Ireland
        # Italian (Italy)
        "Diego": "voice-175",  # Male, Italian, Italy
        "Elsa": "voice-176",  # Female, Italian, Italy
        "Isabella": "voice-178",  # Female, Italian, Italy
        # Japanese (Japan)
        "Keita": "voice-179",  # Male, Japanese, Japan
        "Nanami": "voice-180",  # Female, Japanese, Japan
        # Javanese (Indonesia)
        "Dimas": "voice-181",  # Male, Javanese, Indonesia
        "Siti": "voice-182",  # Female, Javanese, Indonesia
        # Kannada (India)
        "Gagan": "voice-183",  # Male, Kannada, India
        "Sapna": "voice-184",  # Female, Kannada, India
        # Kazakh (Kazakhstan)
        "Aigul": "voice-185",  # Female, Kazakh, Kazakhstan
        "Daulet": "voice-186",  # Male, Kazakh, Kazakhstan
        # Khmer (Cambodia)
        "Piseth": "voice-187",  # Male, Khmer, Cambodia
        "Sreymom": "voice-188",  # Female, Khmer, Cambodia
        # Korean (South Korea)
        "InJoon": "voice-190",  # Male, Korean, South Korea
        "SunHi": "voice-191",  # Female, Korean, South Korea
        # Lao (Laos)
        "Chanthavong": "voice-192",  # Male, Lao, Laos
        "Keomany": "voice-193",  # Female, Lao, Laos
        # Latvian (Latvia)
        "Everita": "voice-194",  # Female, Latvian, Latvia
        "Nils": "voice-195",  # Male, Latvian, Latvia
        # Lithuanian (Lithuania)
        "Leonas": "voice-196",  # Male, Lithuanian, Lithuania
        "Ona": "voice-197",  # Female, Lithuanian, Lithuania
        # Macedonian (North Macedonia)
        "Aleksandar": "voice-198",  # Male, Macedonian, North Macedonia
        "Marija": "voice-199",  # Female, Macedonian, North Macedonia
        # Malay (Malaysia)
        "Osman": "voice-200",  # Male, Malay, Malaysia
        "Yasmin": "voice-201",  # Female, Malay, Malaysia
        # Malayalam (India)
        "Midhun": "voice-202",  # Male, Malayalam, India
        "Sobhana": "voice-203",  # Female, Malayalam, India
        # Maltese (Malta)
        "Grace": "voice-204",  # Female, Maltese, Malta
        "Joseph": "voice-205",  # Male, Maltese, Malta
        # Marathi (India)
        "Aarohi": "voice-206",  # Female, Marathi, India
        "Manohar": "voice-207",  # Male, Marathi, India
        # Mongolian (Mongolia)
        "Bataa": "voice-208",  # Male, Mongolian, Mongolia
        "Yesui": "voice-209",  # Female, Mongolian, Mongolia
        # Nepali (Nepal)
        "Hemkala": "voice-210",  # Female, Nepali, Nepal
        "Sagar": "voice-211",  # Male, Nepali, Nepal
        # Norwegian (Norway)
        "Finn": "voice-212",  # Male, Norwegian, Norway
        "Pernille": "voice-213",  # Female, Norwegian, Norway
        # Pashto (Afghanistan)
        "GulNawaz": "voice-214",  # Male, Pashto, Afghanistan
        "Latifa": "voice-215",  # Female, Pashto, Afghanistan
        # Persian (Iran)
        "Dilara": "voice-216",  # Female, Persian, Iran
        "Farid": "voice-217",  # Male, Persian, Iran
        # Polish (Poland)
        "Marek": "voice-218",  # Male, Polish, Poland
        "Zofia": "voice-219",  # Female, Polish, Poland
        # Portuguese (Brazil)
        "Antonio": "voice-220",  # Male, Portuguese, Brazil
        "Francisca": "voice-221",  # Female, Portuguese, Brazil
        # Portuguese (Portugal)
        "Duarte": "voice-223",  # Male, Portuguese, Portugal
        "Raquel": "voice-224",  # Female, Portuguese, Portugal
        # Romanian (Romania)
        "Alina": "voice-225",  # Female, Romanian, Romania
        "Emil": "voice-226",  # Male, Romanian, Romania
        # Russian (Russia)
        "Dmitry": "voice-227",  # Male, Russian, Russia
        "Svetlana": "voice-228",  # Female, Russian, Russia
        # Serbian (Serbia)
        "Nicholas": "voice-229",  # Male, Serbian, Serbia
        "Sophie": "voice-230",  # Female, Serbian, Serbia
        # Sinhala (Sri Lanka)
        "Sameera": "voice-231",  # Male, Sinhala, Sri Lanka
        "Thilini": "voice-232",  # Female, Sinhala, Sri Lanka
        # Slovak (Slovakia)
        "Lukas": "voice-233",  # Male, Slovak, Slovakia
        "Viktoria": "voice-234",  # Female, Slovak, Slovakia
        # Slovenian (Slovenia)
        "Petra": "voice-235",  # Female, Slovenian, Slovenia
        "Rok": "voice-236",  # Male, Slovenian, Slovenia
        # Somali (Somalia)
        "Muuse": "voice-237",  # Male, Somali, Somalia
        "Ubax": "voice-238",  # Female, Somali, Somalia
        # Sundanese (Indonesia)
        "Jajang": "voice-284",  # Male, Sundanese, Indonesia
        "Tuti": "voice-285",  # Female, Sundanese, Indonesia
        # Swahili (Kenya)
        "Rafiki": "voice-286",  # Male, Swahili, Kenya
        "Zuri": "voice-287",  # Female, Swahili, Kenya
        # Swahili (Tanzania)
        "Daudi": "voice-288",  # Male, Swahili, Tanzania
        "Rehema": "voice-289",  # Female, Swahili, Tanzania
        # Swedish (Sweden)
        "Mattias": "voice-290",  # Male, Swedish, Sweden
        "Sofie": "voice-291",  # Female, Swedish, Sweden
        # Tamil (India)
        "Pallavi": "voice-292",  # Female, Tamil, India
        "Valluvar": "voice-293",  # Male, Tamil, India
        # Tamil (Sri Lanka)
        "Kumar": "voice-294",  # Male, Tamil, Sri Lanka
        "Saranya": "voice-295",  # Female, Tamil, Sri Lanka
        # Tamil (Malaysia)
        "Kani": "voice-296",  # Female, Tamil, Malaysia
        "Surya": "voice-297",  # Male, Tamil, Malaysia
        # Tamil (Singapore)
        "Anbu": "voice-298",  # Male, Tamil, Singapore
        "Venba": "voice-299",  # Female, Tamil, Singapore
        # Telugu (India)
        "Mohan": "voice-300",  # Male, Telugu, India
        "Shruti": "voice-301",  # Female, Telugu, India
        # Thai (Thailand)
        "Niwat": "voice-302",  # Male, Thai, Thailand
        "Premwadee": "voice-303",  # Female, Thai, Thailand
        # Turkish (Turkey)
        "Ahmet": "voice-304",  # Male, Turkish, Turkey
        "Emel": "voice-305",  # Female, Turkish, Turkey
        # Ukrainian (Ukraine)
        "Ostap": "voice-306",  # Male, Ukrainian, Ukraine
        "Polina": "voice-307",  # Female, Ukrainian, Ukraine
        # Urdu (India)
        "Gul": "voice-308",  # Female, Urdu, India
        "Salman": "voice-309",  # Male, Urdu, India
        # Urdu (Pakistan)
        "Asad": "voice-310",  # Male, Urdu, Pakistan
        "Uzma": "voice-311",  # Female, Urdu, Pakistan
        # Uzbek (Uzbekistan)
        "Madina": "voice-312",  # Female, Uzbek, Uzbekistan
        "Sardor": "voice-313",  # Male, Uzbek, Uzbekistan
        # Vietnamese (Vietnam)
        "HoaiMy": "voice-314",  # Female, Vietnamese, Vietnam
        "NamMinh": "voice-315",  # Male, Vietnamese, Vietnam
        # Welsh (United Kingdom)
        "Aled": "voice-316",  # Male, Welsh, United Kingdom
        "Nia": "voice-317",  # Female, Welsh, United Kingdom
        # Zulu (South Africa)
        "Thando": "voice-318",  # Female, Zulu, South Africa
        "Themba": "voice-319",  # Male, Zulu, South Africa
    }

    def __init__(self, timeout: int = 20, proxies: dict = None):
        """Initializes the SpeechMa TTS client."""
        super().__init__()
        self.api_url = "https://speechma.com/com.api/tts-api.php"
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if proxies:
            self.session.proxies.update(proxies)
        self.timeout = timeout

    def tts(self, text: str, voice: str = "Emma", pitch: int = 0, rate: int = 0) -> str:
        """
        Converts text to speech using the SpeechMa API and saves it to a file.

        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use for TTS (default: "Emma")
            pitch (int): Voice pitch adjustment (-10 to 10, default: 0)
            rate (int): Voice rate/speed adjustment (-10 to 10, default: 0)

        Returns:
            str: Path to the generated audio file

        Raises:
            exceptions.FailedToGenerateResponseError: If there is an error generating or saving the audio.
        """
        assert (
            voice in self.all_voices
        ), f"Voice '{voice}' not one of [{', '.join(self.all_voices.keys())}]"

        filename = pathlib.Path(tempfile.mktemp(suffix=".mp3", dir=self.temp_dir))
        voice_id = self.all_voices[voice]

        # Prepare payload for the job-based API
        payload = {
            "text": text,
            "voice": voice_id,
            "pitch": pitch,
            "rate": rate,
            "volume": 100
        }

        try:
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            resp_json = response.json()
            if not resp_json.get("success") or "data" not in resp_json or "job_id" not in resp_json["data"]:
                raise exceptions.FailedToGenerateResponseError(f"SpeechMa API error: {resp_json}")
            job_id = resp_json["data"]["job_id"]

            # Poll for job completion
            status_url = f"https://speechma.com/com.api/tts-api.php/status/{job_id}"
            for _ in range(30):  # up to ~30 seconds
                status_resp = self.session.get(
                    status_url,
                    headers=self.headers,
                    timeout=self.timeout
                )
                status_resp.raise_for_status()
                status_json = status_resp.json()
                if status_json.get("success") and status_json.get("data", {}).get("status") == "completed":
                    break
                time.sleep(1)
            else:
                raise exceptions.FailedToGenerateResponseError("TTS job did not complete in time.")

            # Download the audio file (API provides a URL in the status response)
            data = status_json["data"]
            audio_url = f"https://speechma.com/com.api/tts-api.php/audio/{job_id}"
            audio_resp = self.session.get(audio_url, timeout=self.timeout)
            audio_resp.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(audio_resp.content)
            return filename.as_posix()

        except requests.exceptions.RequestException as e:
            raise exceptions.FailedToGenerateResponseError(
                f"Failed to perform the operation: {e}"
            )

# Example usage
if __name__ == "__main__":
    speechma = SpeechMaTTS()
    text = "This is a test of the SpeechMa text-to-speech API. It supports multiple sentences."
    audio_file = speechma.tts(text, voice="Emma")
    print(f"Audio saved to: {audio_file}")
