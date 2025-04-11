import re
import requests
import urllib.parse
from bs4 import BeautifulSoup

# Get user input for the query
user_query = input(">>> ")

# Encode the user query for URL
encoded_query = urllib.parse.quote_plus(user_query)

# Construct the URL with the user's query
url = f"https://iask.ai/?mode=question&options[detail_level]=detailed&q={encoded_query}"

# Make a request
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find the main div with class where the text content is present
main_div = soup.find("div", class_="max-w-full prose leading-[26px] text-[#111827] no-overflow-anchoring")

# Extract all elements of interest
output_lines = []


if main_div:
    for child in main_div.children:
        if child.name == "h1" or child.name == "h2" or child.name == "h3":
            output_lines.append(f"\n**{child.text.strip()}**\n")
        elif child.name == "p":
            text = child.text.strip()
            text = re.sub(r"^According to Ask AI & Question AI www\\.iAsk\\.ai:\\s*", "", text).strip()
            output_lines.append(text + "\n")
        elif child.name == "ol" or child.name == "ul":
            for li in child.find_all("li"):
                output_lines.append("- " + li.text.strip() + "\n")
        elif child.name == "div" and "footnotes" in child.get("class", []):
            output_lines.append("\n**Authoritative Sources**\n")
            for li in child.find_all("li"):
                link = li.find("a")
                if link:
                    output_lines.append(f"- {link.text.strip()} ({link['href']})\n")


final_output = "\n".join(output_lines)

# Remove the attribution text at the end
final_output = re.sub(r"\s*Answer Provided by\s*www\.iAsk\.ai\s*– Ask AI\.\s*$", "", final_output)

print(final_output)