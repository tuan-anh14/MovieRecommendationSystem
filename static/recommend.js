$(function () {
  // Button will be disabled until we type anything inside the input field
  const source = document.getElementById('autoComplete');
  const inputHandler = function (e) {
    if (e.target.value == "") {
      $('.movie-button').attr('disabled', true);
    }
    else {
      $('.movie-button').attr('disabled', false);
    }
  }
  source.addEventListener('input', inputHandler);

  $('.movie-button').on('click', function () {
    $('#loader').fadeIn();
    var my_api_key = '4f7c3ea95b05ea5a83b661924a0c10ee';
    var rec_mode = $('input[name="rec_mode"]:checked').val();
    var title = $('.movie').val();
    var desc = $('#desc-input').val();
    if (rec_mode === 'desc') {
      if (!desc.trim()) {
        alert('Vui lòng nhập mô tả phim!');
        $('#loader').fadeOut();
        return;
      }
      load_details(my_api_key, desc, rec_mode);
    } else {
      if (title == "") {
        $('.results').css('display', 'none');
        $('.fail').css('display', 'block');
        $('#loader').fadeOut();
      }
      else {
        load_details(my_api_key, title, rec_mode);
      }
    }
  });
});

// will be invoked when clicking on the recommended movies
function recommendcard(e) {
  var my_api_key = '4f7c3ea95b05ea5a83b661924a0c10ee';
  var title = e.getAttribute('title');
  load_details(my_api_key, title);
}

// get the basic details of the movie from the API (based on the name of the movie)
function load_details(my_api_key, title, rec_mode) {
  $.ajax({
    type: 'GET',
    url: 'https://api.themoviedb.org/3/search/movie?api_key=' + my_api_key + '&query=' + title,
    success: function (movie) {
      if (movie.results.length < 1) {
        $('.fail').css('display', 'block');
        $('.results').css('display', 'none');
        $('#loader').fadeOut();
      }
      else {
        $('.fail').css('display', 'none');
        $('.results').css('display', 'block');
        var movie_id = movie.results[0].id;
        var movie_title = movie.results[0].original_title;
        movie_recs(movie_title, movie_id, my_api_key, rec_mode);
      }
    },
    error: function () {
      $('#loader').fadeOut();
      alert('Invalid Request');
    },
  });
}

// passing the movie name to get the similar movies from python's flask
function movie_recs(movie_title, movie_id, my_api_key, rec_mode) {
  $.ajax({
    type: 'POST',
    url: "/similarity",
    data: { 'name': movie_title, 'rec_mode': rec_mode },
    success: function (recs) {
      if (recs.startsWith("Sorry!") || recs.startsWith("Không tìm thấy") || recs.startsWith("Chế độ gợi ý") || recs.startsWith("Không hỗ trợ")) {
        $('.fail').text(recs).css('display', 'block');
        $('.results').css('display', 'none');
        $('#loader').fadeOut();
      }
      else {
        $('.fail').css('display', 'none');
        $('.results').css('display', 'block');
        var movie_arr = recs.split('---');
        var arr = [];
        for (const movie in movie_arr) {
          arr.push(movie_arr[movie]);
        }
        get_movie_details(movie_id, my_api_key, arr, movie_title);
      }
    },
    error: function () {
      $('#loader').fadeOut();
      alert("error recs");
    },
  });
}

// get all the details of the movie using the movie id.
function get_movie_details(movie_id, my_api_key, arr, movie_title) {
  $.ajax({
    type: 'GET',
    url: 'https://api.themoviedb.org/3/movie/' + movie_id + '?api_key=' + my_api_key,
    success: function (movie_details) {
      show_details(movie_details, arr, movie_title, my_api_key, movie_id);
    },
    error: function () {
      $('#loader').fadeOut();
      alert("API Error!");
    },
  });
}

// passing all the details to python's flask for displaying and scraping the movie reviews using imdb id
async function show_details(movie_details, arr, movie_title, my_api_key, movie_id) {
  var imdb_id = movie_details.imdb_id;
  var poster = 'https://image.tmdb.org/t/p/original' + movie_details.poster_path;
  var overview = movie_details.overview;
  var genres = movie_details.genres;
  var rating = movie_details.vote_average;
  var vote_count = movie_details.vote_count;
  var release_date = new Date(movie_details.release_date);
  var runtime = parseInt(movie_details.runtime);
  var status = movie_details.status;
  var genre_list = [];
  for (var genre of genres) {
    genre_list.push(genre.name);
  }
  var my_genre = genre_list.join(", ");
  if (runtime % 60 == 0) {
    runtime = Math.floor(runtime / 60) + " hour(s)"
  }
  else {
    runtime = Math.floor(runtime / 60) + " hour(s) " + (runtime % 60) + " min(s)"
  }
  // Lấy poster, cast, cast detail bất đồng bộ
  const arr_poster = await get_movie_posters(arr, my_api_key);
  const movie_cast = await get_movie_cast(movie_id, my_api_key);
  const ind_cast = await get_individual_cast(movie_cast, my_api_key);

  var details = {
    'title': movie_title,
    'cast_ids': JSON.stringify(movie_cast.cast_ids),
    'cast_names': JSON.stringify(movie_cast.cast_names),
    'cast_chars': JSON.stringify(movie_cast.cast_chars),
    'cast_profiles': JSON.stringify(movie_cast.cast_profiles),
    'cast_bdays': JSON.stringify(ind_cast.cast_bdays),
    'cast_bios': JSON.stringify(ind_cast.cast_bios),
    'cast_places': JSON.stringify(ind_cast.cast_places),
    'imdb_id': imdb_id,
    'poster': poster,
    'genres': my_genre,
    'overview': overview,
    'rating': rating,
    'vote_count': vote_count.toLocaleString(),
    'release_date': release_date.toDateString().split(' ').slice(1).join(' '),
    'runtime': runtime,
    'status': status,
    'rec_movies': JSON.stringify(arr),
    'rec_posters': JSON.stringify(arr_poster),
  };

  $.ajax({
    type: 'POST',
    data: details,
    url: "/recommend",
    dataType: 'html',
    complete: function () {
      $('#loader').fadeOut();
    },
    success: function (response) {
      $('.results').html(response);
      $('#autoComplete').val('');
      $(window).scrollTop(0);
    }
  });
}

// Thay thế các hàm lấy dữ liệu cast, poster, cast detail bằng Promise
async function get_movie_posters(arr, my_api_key) {
  const arr_poster_list = [];
  for (const m of arr) {
    try {
      const m_data = await $.ajax({
        type: 'GET',
        url: 'https://api.themoviedb.org/3/search/movie?api_key=' + my_api_key + '&query=' + m,
      });
      if (m_data.results && m_data.results[0] && m_data.results[0].poster_path) {
        arr_poster_list.push('https://image.tmdb.org/t/p/original' + m_data.results[0].poster_path);
      } else {
        arr_poster_list.push('');
      }
    } catch (e) {
      arr_poster_list.push('');
    }
  }
  return arr_poster_list;
}

async function get_movie_cast(movie_id, my_api_key) {
  const cast_ids = [];
  const cast_names = [];
  const cast_chars = [];
  const cast_profiles = [];
  try {
    const my_movie = await $.ajax({
      type: 'GET',
      url: "https://api.themoviedb.org/3/movie/" + movie_id + "/credits?api_key=" + my_api_key,
    });
    let top_cast = [];
    if (my_movie.cast && my_movie.cast.length >= 10) {
      top_cast = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    } else {
      top_cast = [0, 1, 2, 3, 4];
    }
    for (const idx of top_cast) {
      if (my_movie.cast[idx]) {
        cast_ids.push(my_movie.cast[idx].id);
        cast_names.push(my_movie.cast[idx].name);
        cast_chars.push(my_movie.cast[idx].character);
        cast_profiles.push("https://image.tmdb.org/t/p/original" + my_movie.cast[idx].profile_path);
      }
    }
  } catch (e) { }
  return { cast_ids, cast_names, cast_chars, cast_profiles };
}

async function get_individual_cast(movie_cast, my_api_key) {
  const cast_bdays = [];
  const cast_bios = [];
  const cast_places = [];
  for (const cast_id of movie_cast.cast_ids) {
    try {
      const cast_details = await $.ajax({
        type: 'GET',
        url: 'https://api.themoviedb.org/3/person/' + cast_id + '?api_key=' + my_api_key,
      });
      cast_bdays.push((new Date(cast_details.birthday)).toDateString().split(' ').slice(1).join(' '));
      cast_bios.push(cast_details.biography);
      cast_places.push(cast_details.place_of_birth);
    } catch (e) {
      cast_bdays.push('');
      cast_bios.push('');
      cast_places.push('');
    }
  }
  return { cast_bdays, cast_bios, cast_places };
}
