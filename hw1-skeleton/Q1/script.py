# TODO(BENB): DELETE_THIS: api_key 62cbc6a08b82a26943ea3b1740ab6dd9

# import collections
import http.client
import json
import sys
import time

ceil_divide = lambda num, dom: int(-(-num // float(dom)))


def retrieve_api_key(argv):
    if len(argv) < 1:
        print('API_KEY required; i.e. script.py <API_KEY>')
        sys.exit(2)
    else:
        api_key = argv[0]

    return api_key


def RateLimited(maxPerSecond):
    # TODO (BEN B): Rewrite this
    minInterval = 1.0 / float(maxPerSecond)

    def decorate(func):
        lastTimeCalled = [0.0]

        def rateLimitedFunction(*args, **kwargs):
            elapsed = time.time() - lastTimeCalled[0]
            leftToWait = minInterval - elapsed
            if leftToWait > 0:
                time.sleep(leftToWait)
            ret = func(*args, **kwargs)
            lastTimeCalled[0] = time.time()
            return ret

        return rateLimitedFunction

    return decorate


@RateLimited(3)
def request_page(api_key, page):
    conn = http.client.HTTPSConnection("api.themoviedb.org")

    payload = "{}"

    conn.request("GET",
                 "/3/discover/movie?sort_by=popularity.desc&with_genres=18&primary_release_date.gte=2004&page={}&api_key={}"
                 .format(page, api_key),
                 payload
                 )

    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    conn.close()

    return data


def write_popular_movies_to_csv(n, api_key):
    count = 0
    num_results_per_page = 20
    num_request = ceil_divide(n, num_results_per_page)

    f = open('movie_ID_name.csv', 'w')
    for page in range(1, num_request + 1):
        data = request_page(api_key, page)
        results = data['results']
        for result in range(len(results)):
            id = str(data['results'][result]['id'])
            title = data['results'][result]['title']
            if count >= n:
                return
            elif count == n-1:
                f.write(id + ',' + title.encode('utf-8'))
            else:
                f.write(id + ',' + title.encode('utf-8') + '\n')
            count = count + 1
    f.close()

def main(argv):
    start = time.time()
    api_key = retrieve_api_key(argv)
    write_popular_movies_to_csv(350, api_key)
    end = time.time()
    print('script executed in ' + str((end - start)) + ' secs')


if __name__ == "__main__":
    main(sys.argv[1:])
