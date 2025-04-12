class User {
  constructor(cognitoUser) {
    this.attributes = {};
    if (cognitoUser && cognitoUser.UserAttributes) {
      cognitoUser.UserAttributes.forEach(attr => {
        this.attributes[attr.Name] = attr.Value;
      });
    }
  }

  toJSON() {
    return {
      username: (this.attributes.preferred_username || this.attributes.email || '').replace(/^Dr\.?\s*/i, ''),
      email: this.attributes.email,
      sub: this.attributes.sub,
      lastLogin: new Date().toISOString(),
      ...this.attributes
    };
  }

  static fromCognitoUser(cognitoUser) {
    return new User(cognitoUser);
  }
}

module.exports = User; 