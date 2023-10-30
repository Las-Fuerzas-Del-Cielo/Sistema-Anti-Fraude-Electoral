import { rateLimit } from 'express-rate-limit';
import {RATE_LIMITER_WINDOW_MINUTES, RATE_LIMITER_REQUEST_PER_WINDOW} from "../constants";

const rateLimiter = rateLimit({
    windowMs: RATE_LIMITER_WINDOW_MINUTES * 60 * 1000,
    limit: RATE_LIMITER_REQUEST_PER_WINDOW,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
});

export default rateLimiter;
