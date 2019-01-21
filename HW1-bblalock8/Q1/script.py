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
def request_tmdb_api(request):
    conn = http.client.HTTPSConnection("api.themoviedb.org")

    payload = "{}"

    conn.request("GET",
                 request,
                 payload
                 )

    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    conn.close()

    return data


def request_n_most_popular_dramas(api_key, page):
    return request_tmdb_api(
        "/3/discover/movie?sort_by=popularity.desc&with_genres=18&primary_release_date.gte=2004&page={}&api_key={}".format(
            page, api_key))


def write_tuple_list_to_csv(t_list, file_name):
    f = open(file_name, 'w')
    num_tups = len(t_list)

    count = 0
    for tup in t_list:
        if count == num_tups - 1:
            f.write(str(tup[0]) + ',' + str(tup[1]))
            f.close()
            return
        else:
            f.write(str(tup[0]) + ',' + str(tup[1]) + '\n')
        count = count + 1
    f.close()


def collect_n_popular_movies(n, api_key):
    count = 0
    num_results_per_page = 20
    num_request = ceil_divide(n, num_results_per_page)

    results = []
    for page in range(1, num_request + 1):
        data = request_n_most_popular_dramas(api_key, page)
        movie_list = data['results']
        for result in range(len(movie_list)):
            id = data['results'][result]['id']
            title = data['results'][result]['title']
            results = results + [(id, title)]
            count = count + 1
            if count >= n:
                return results
    return results


def write_popular_movies_to_csv(n, api_key):
    t_list = collect_n_popular_movies(n, api_key)
    write_tuple_list_to_csv(t_list, 'movie_ID_name.csv')


def request_n_similar_movies(n, movie_id, api_key):
    num_results_per_page = 20
    num_request = ceil_divide(n, num_results_per_page)

    results = []
    for page in range(1, num_request + 1):
        data = request_tmdb_api(
            'https://api.themoviedb.org/3/movie/{}/similar?api_key={}&page={}'.format(movie_id, api_key, page))
        results = results + [data['results'][i]['id'] for i in range(0, len(data['results']))]
    results = [(movie_id, results[i]) for i in range(0, len(results)) if i < n]
    print(results)
    print(len(results))

    # 'https://api.themoviedb.org/3/movie/{}/similar?api_key={}&page={}'.format(movie_id, api_key, page)


def main(argv):
    start = time.time()
    api_key = retrieve_api_key(argv)
    write_popular_movies_to_csv(350, api_key)
    # request_n_similar_movies(5, 424694, api_key)
    end = time.time()
    print('script executed in ' + str((end - start)) + ' secs')


if __name__ == "__main__":
    main(sys.argv[1:])
