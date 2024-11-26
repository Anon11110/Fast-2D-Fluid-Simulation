bool IsObstacle(vec2 uv)
{
    vec2 clamped_uv = clamp(uv, 0.0, 1.0);
    float obstacle = texture(obstacle_mask_texture, clamped_uv).r;
    return obstacle > 0.5;
}

vec2 CalculateNormal(vec2 uv)
{
    vec2 dX = vec2(1.0 / push_constants.texture_width, 0.0);
    vec2 dY = vec2(0.0, 1.0 / push_constants.texture_height);

    float dx = texture(obstacle_mask_texture, uv + dX).r - texture(obstacle_mask_texture, uv - dX).r;
    float dy = texture(obstacle_mask_texture, uv + dY).r - texture(obstacle_mask_texture, uv - dY).r;

    return normalize(vec2(dx, dy));
}

vec2 ApplySlipperyBoundary(vec2 uv, vec2 velocity)
{
    if (!IsObstacle(uv))
        return velocity;

    vec2 normal = CalculateNormal(uv);
    float normal_component = dot(velocity, normal);
    vec2 tangential_component = velocity - normal_component * normal;

    return clamp(tangential_component, vec2(0, 0), velocity);
}
