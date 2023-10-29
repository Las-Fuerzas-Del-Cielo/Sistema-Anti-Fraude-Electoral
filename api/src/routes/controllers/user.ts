export const createUser:Function = function(result){
  // Mocked Logic
  return result({ message: 'User created', user: this.body })
};
export const getUserRoles:Function = function(result){
  // Mocked Logic
  const roles = ['admin', 'user']
  return result({ roles })
};
export const getUser:Function = function(result){
  // Mocked Logic
  return result({ user: 'John Doe' })
};
export const getUsers:Function = function(result){
  // Mocked Logic
  return result([{ userId: 1 }, { userId: 2 }])
};