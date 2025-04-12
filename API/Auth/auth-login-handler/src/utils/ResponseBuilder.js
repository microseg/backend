class ResponseBuilder {
  static success(data = {}, statusCode = 200) {
    return {
      statusCode,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Credentials': true
      },
      body: JSON.stringify({
        success: true,
        data
      })
    };
  }

  static error(message, statusCode = 400) {
    return {
      statusCode,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Credentials': true
      },
      body: JSON.stringify({
        success: false,
        error: {
          message
        }
      })
    };
  }
}

module.exports = ResponseBuilder; 