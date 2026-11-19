def response_checker_no_thinking_strict(response):
    response = response.strip()
    return response == "<judgment>YES</judgment>" or response == "<judgment>NO</judgment>"


def response_checker_no_thinking_lenient(response):
    response = response.strip()
    return response.endswith("<judgment>YES</judgment>") or response.endswith("<judgment>NO</judgment>")


def response_checker_thinking_strict(response):
    response = response.strip()
    return response.endswith("</think>\n\n<judgment>YES</judgment>") or response.endswith("</think>\n\n<judgment>NO</judgment>")


def response_checker_thinking_lenient(response):
    response = response.strip()
    return response.endswith("<judgment>YES</judgment>") or response.endswith("<judgment>NO</judgment>")

