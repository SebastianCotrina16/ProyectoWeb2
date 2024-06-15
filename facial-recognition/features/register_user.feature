Feature: Register user
  As a user
  I want to register with my details and photo
  So that I can be recognized later

  Scenario: Successfully register a user
    Given I have a user with DNI "12345678" and name "John Doe"
    When I upload the registration image "Fotos/SebastianCotrina.png"
    Then the user should be successfully registered
    And the user with DNI "12345678" should exist in the database
    And the registered user name should be "John Doe"
