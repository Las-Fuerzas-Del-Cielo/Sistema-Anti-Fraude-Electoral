import type { Express } from 'express'
import userRoutes from './user'
import mesaRoutes from './mesa'
import denunciaRoutes from './denuncia'
import fiscalizarRoutes from './fiscalizar'

export function registerRoutes(app: Express) {
  app.use('/api/user', userRoutes)
  app.use('/api/mesa', mesaRoutes)
  app.use('/api/denuncia', denunciaRoutes)
  app.use('/api/fiscalizar', fiscalizarRoutes)
  return app
}
