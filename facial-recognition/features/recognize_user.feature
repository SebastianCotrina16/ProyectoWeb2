Feature: Recognize user
  As a user
  I want to be recognized by my photo
  So that I can verify my identity

  Scenario: Successfully recognize a registered user
    Given a user with DNI "12345678" and name "John Doe" is registered with image "Fotos/Keanu.webp"
    When I upload the recognition image "Fotos/Keanu2.webp"
    Then the user with DNI "12345678" should be recognized
    And the recognized user name should be "John Doe"

  Scenario: Fail to recognize an unknown user
    When I upload the recognition image "Fotos/Adam.jpg"
    Then no user should be recognized
